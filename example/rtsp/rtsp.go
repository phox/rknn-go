package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	rknnlite "github.com/phox/rknn-go"
	"github.com/phox/rknn-go/postprocess"
	"github.com/phox/rknn-go/preprocess"
	"github.com/phox/rknn-go/render"
	"github.com/phox/rknn-go/tracker"
	"gocv.io/x/gocv"
)

var (
	// FPS is the number of FPS to simulate
	FPS         = 30
	FPSinterval = time.Duration(float64(time.Second) / float64(FPS))

	clrBlack = color.RGBA{R: 0, G: 0, B: 0, A: 255}
	clrWhite = color.RGBA{R: 255, G: 255, B: 255, A: 255}
)

// Timing is a struct to hold timers used for finding execution time
// for various parts of the process
type Timing struct {
	ProcessStart       time.Time
	DetObjStart        time.Time
	DetObjInferenceEnd time.Time
	DetObjEnd          time.Time
	TrackerStart       time.Time
	TrackerEnd         time.Time
	RenderingStart     time.Time
	ProcessEnd         time.Time
}

// ResultFrame is a struct to wrap the gocv byte buffer and error result
type ResultFrame struct {
	Buf *gocv.NativeByteBuffer
	Err error
}

// YOLOProcessor defines an interface for different versions of YOLO
// models used for object detection
type YOLOProcessor interface {
	DetectObjects(outputs *rknnlite.Outputs,
		resizer *preprocess.Resizer) postprocess.DetectionResult
}

// RTSPSource defines the RTSP video source to use for playback.
type RTSPSource struct {
	URL      string
	Codec    string // h264 or h265
	Username string
	Password string
}

// GetRTSPURL returns the full RTSP URL with authentication if provided
func (r *RTSPSource) GetRTSPURL() string {
	if r.Username != "" && r.Password != "" {
		// Parse the URL to insert authentication
		regex := regexp.MustCompile(`^rtsp://(.*)$`)
		if regex.MatchString(r.URL) {
			return fmt.Sprintf("rtsp://%s:%s@%s", r.Username, r.Password, regex.ReplaceAllString(r.URL, "$1"))
		}
	}
	return r.URL
}

// Demo defines the struct for running the object tracking demo
type Demo struct {
	// rtspSrc holds details on our RTSP video source for playback
	rtspSrc *RTSPSource
	// pool of rknnlite runtimes to perform inference in parallel
	pool *rknnlite.Pool
	// process is a YOLO object detection processor
	process YOLOProcessor
	// labels are the COCO labels the YOLO model was trained on
	labels []string
	// limitObjs restricts object detection results to be only those provided
	limitObjs []string
	// resizer handles scaling of source image to input tensors
	resizer *preprocess.Resizer
	// modelType is the type of YOLO model to use as processor that was passed
	// as a command line flag
	modelType string
	// renderFormat indicates which rendering type to use with instance
	// segmentation, outline or mask
	renderFormat string
	// useTracking indicates whether to use ByteTrack for object tracking
	useTracking bool
	// frameBuffer 用于缓存视频帧，减少卡顿
	frameBuffer chan gocv.Mat
	// bufferSize 缓冲区大小（帧数）
	bufferSize int
	// rtspRunning 标记RTSP流是否正在运行
	rtspRunning bool
	// rtspMutex 用于保护rtspRunning和clientCount的互斥锁
	rtspMutex sync.Mutex
	// clientCount 当前连接的客户端数量
	clientCount int
	// rtspExitCh 用于关闭RTSP流的通道
	rtspExitCh chan struct{}
	// 用于保护frameBuffer的互斥锁
	frameBufferMutex sync.Mutex
}

// NewDemo returns and instance of Demo, a streaming HTTP server showing
// video with object detection
func NewDemo(rtspSrc *RTSPSource, modelFile, labelFile string, poolSize int,
	modelType string, renderFormat string, cores []rknnlite.CoreMask) (*Demo, error) {

	var err error

	d := &Demo{
		rtspSrc:     rtspSrc,
		limitObjs:   make([]string, 0),
		clientCount: 0,
		rtspRunning: false,
		bufferSize:  90, // 默认3秒缓冲（30fps）
	}

	// create new pool
	d.pool, err = rknnlite.NewPool(poolSize, modelFile, cores)

	if err != nil {
		log.Fatalf("Error creating RKNN pool: %v\n", err)
	}

	// set runtime to leave output tensors as int8
	d.pool.SetWantFloat(false)

	// create resizer to handle scaling of input image to inference tensor
	// input size requirements
	rt := d.pool.Get()

	// 初始化resizer，暂时使用默认值，后续会在打开RTSP流后更新
	d.resizer = preprocess.NewResizer(640, 480,
		int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))

	d.pool.Return(rt)

	// 尝试连接RTSP流以获取实际分辨率
	vcap, err := gocv.OpenVideoCapture(rtspSrc.GetRTSPURL())
	if err == nil {
		defer vcap.Close()

		// 获取视频流的实际宽高
		width := int(vcap.Get(gocv.VideoCaptureFrameWidth))
		height := int(vcap.Get(gocv.VideoCaptureFrameHeight))

		if width > 0 && height > 0 {
			// 更新resizer使用实际分辨率
			d.resizer = preprocess.NewResizer(width, height,
				int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
			log.Printf("RTSP stream resolution: %dx%d\n", width, height)
		}
	} else {
		log.Printf("Warning: Could not connect to RTSP stream to get resolution: %v\n", err)
		log.Printf("Using default resolution 640x480 for resizer\n")
	}

	// create YOLO post processor
	switch modelType {
	case "v8":
		d.process = postprocess.NewYOLOv8(postprocess.YOLOv8COCOParams())
	case "v5":
		d.process = postprocess.NewYOLOv5(postprocess.YOLOv5COCOParams())
	case "v10":
		d.process = postprocess.NewYOLOv10(postprocess.YOLOv10COCOParams())
	case "v11":
		d.process = postprocess.NewYOLOv11(postprocess.YOLOv11COCOParams())
	case "x":
		d.process = postprocess.NewYOLOX(postprocess.YOLOXCOCOParams())
	case "v5seg":
		d.process = postprocess.NewYOLOv5Seg(postprocess.YOLOv5SegCOCOParams())
		// force FPS to 10, as we don't have enough CPU power to do 30 FPS
		FPS = 10
		FPSinterval = time.Duration(float64(time.Second) / float64(FPS))
		log.Println("***WARNING*** Instance Segmentation requires a lot of CPU, downgraded to 10 FPS")
	case "v8seg":
		d.process = postprocess.NewYOLOv8Seg(postprocess.YOLOv8SegCOCOParams())
		// force FPS to 10, as we don't have enough CPU power to do 30 FPS
		FPS = 10
		FPSinterval = time.Duration(float64(time.Second) / float64(FPS))
		log.Println("***WARNING*** Instance Segmentation requires a lot of CPU, downgraded to 10 FPS")
	case "v8pose":
		d.process = postprocess.NewYOLOv8Pose(postprocess.YOLOv8PoseCOCOParams())
	case "v8obb":
		d.process = postprocess.NewYOLOv8obb(postprocess.YOLOv8obbDOTAv1Params())
	default:
		log.Fatal("Unknown model type, use 'v5', 'v8', 'v10', 'v11', 'x', 'v5seg', 'v8seg', 'v8pose', or 'v8obb'")
	}

	d.modelType = modelType
	d.renderFormat = renderFormat

	// load in Model class names
	d.labels, err = rknnlite.LoadLabels(labelFile)

	if err != nil {
		return nil, fmt.Errorf("Error loading model labels: %w", err)
	}

	return d, nil
}

// LimitObjects limits the object detection kind to the labels provided, eg:
// limit to just "person".  Provide a comma delimited list of labels to
// restrict to.
func (d *Demo) LimitObjects(lim string) {

	words := strings.Split(lim, ",")

	for _, word := range words {
		trimmed := strings.TrimSpace(word)

		// check if word is an actual label in our labels file
		if containsStr(d.labels, trimmed) {
			d.limitObjs = append(d.limitObjs, trimmed)
		}
	}

	log.Printf("Limiting object detection class to: %s\n", strings.Join(d.limitObjs, ", "))
}

// containsStr is a function that takes a string slice and checks if a given
// string exists in the slice
func containsStr(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}

	return false
}

// Stream is the HTTP handler function used to stream video frames to browser
func (d *Demo) Stream(w http.ResponseWriter, r *http.Request) {
	log.Printf("新客户端连接已建立")

	// 增加客户端计数并检查RTSP流状态
	d.rtspMutex.Lock()
	d.clientCount++
	clientID := d.clientCount
	rtspRunning := d.rtspRunning
	log.Printf("当前客户端数量: %d", d.clientCount)

	// 如果RTSP流未运行，则启动它
	if !rtspRunning {
		// 初始化帧缓冲区（如果尚未初始化）
		if d.frameBuffer == nil {
			d.frameBufferMutex.Lock()
			if d.frameBuffer == nil { // 双重检查
				// 设置默认缓冲区大小（如果未设置）
				if d.bufferSize <= 0 {
					d.bufferSize = 90 // 默认3秒缓冲（30fps）
				}
				d.frameBuffer = make(chan gocv.Mat, d.bufferSize)
				log.Printf("已创建帧缓冲区，大小: %d 帧", d.bufferSize)
			}
			d.frameBufferMutex.Unlock()
		}

		// 创建退出通道
		if d.rtspExitCh == nil {
			d.rtspExitCh = make(chan struct{})
		}

		// 标记RTSP流为运行状态
		d.rtspRunning = true
		log.Printf("启动RTSP流")

		// 启动RTSP流读取
		go d.startRTSPStream(d.frameBuffer, d.rtspExitCh)
	}
	d.rtspMutex.Unlock()

	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")

	// 创建一个bytetracker用于跟踪检测到的对象
	byteTrack := tracker.NewBYTETracker(FPS, FPS*10, 0.5, 0.6, 0.8)

	// 创建轨迹历史记录
	trail := tracker.NewTrail(90)

	// 创建用于注释图像的Mat
	resImg := gocv.NewMat()
	defer resImg.Close()

	// 用于计算FPS
	frameCount := 0
	startTime := time.Now()
	fps := float64(0)

	// 创建通道以接收处理后的帧
	recvFrame := make(chan ResultFrame, 30)

	// 用于控制输出帧率的计时器
	outputTicker := time.NewTicker(time.Second / time.Duration(FPS))
	defer outputTicker.Stop()

	// 用于定期打印缓冲区状态的计时器
	statsTicker := time.NewTicker(5 * time.Second)
	defer statsTicker.Stop()

	// 用于检测缓冲区状态的计时器
	bufferCheckTicker := time.NewTicker(500 * time.Millisecond)
	defer bufferCheckTicker.Stop()

	frameNum := 0
	// 标记是否有处理中的帧
	processingFrame := false
	// 记录处理延迟
	processDelay := time.Duration(0)
	// 记录缓冲区状态
	bufferStatus := "Low"
	// 记录最后一次输出时间，用于自适应帧率控制
	lastOutputTime := time.Now()
	// 自适应输出间隔，根据处理延迟动态调整
	adaptiveInterval := time.Second / time.Duration(FPS)

	log.Printf("客户端 #%d: 开始视频流处理，缓冲区大小: %d 帧 (%.1f 秒)",
		clientID, d.bufferSize, float64(d.bufferSize)/float64(FPS))

	// 预热缓冲区，等待一定数量的帧被缓存后再开始处理
	preWarmSize := d.bufferSize / 2
	if preWarmSize < 1 {
		preWarmSize = 1
	}

	log.Printf("客户端 #%d: 预热缓冲区，等待 %d 帧被缓存...", clientID, preWarmSize)
	for len(d.frameBuffer) < preWarmSize {
		time.Sleep(100 * time.Millisecond)
		// 检查客户端是否已断开连接
		select {
		case <-r.Context().Done():
			log.Printf("客户端 #%d: 在预热阶段断开连接", clientID)
			// 减少客户端计数
			d.handleClientDisconnect(clientID)
			return
		default:
			// 继续等待
		}
	}
	log.Printf("客户端 #%d: 缓冲区预热完成，开始处理视频流", clientID)

loop:
	for {
		select {
		case <-r.Context().Done():
			log.Printf("客户端 #%d: 断开连接", clientID)
			// 减少客户端计数并可能关闭RTSP流
			d.handleClientDisconnect(clientID)
			break loop

		// 定期检查缓冲区状态
		case <-bufferCheckTicker.C:
			// 根据缓冲区状态调整输出帧率
			bufferLen := len(d.frameBuffer)
			bufferCap := cap(d.frameBuffer)

			// 根据缓冲区占用比例调整输出间隔
			if bufferLen < bufferCap/4 {
				// 缓冲区较空，降低输出帧率以积累更多帧
				adaptiveInterval = time.Second / time.Duration(FPS) * 3 / 2
				bufferStatus = "Low"
			} else if bufferLen > bufferCap*3/4 {
				// 缓冲区较满，提高输出帧率以消耗更多帧
				adaptiveInterval = time.Second / time.Duration(FPS) * 2 / 3
				bufferStatus = "High"
			} else {
				// 缓冲区状态正常，使用标准帧率
				adaptiveInterval = time.Second / time.Duration(FPS)
				bufferStatus = "Middle"
			}

			// 根据处理延迟进一步调整输出间隔
			if processDelay > adaptiveInterval {
				// 处理时间超过了帧间隔，适当降低输出帧率
				adaptiveInterval = processDelay * 5 / 4
			}

		// 定期打印缓冲区状态
		case <-statsTicker.C:
			log.Printf("客户端 #%d: 缓冲区状态: %d/%d 帧 (%s), 处理延迟: %.2f ms, 输出帧率: %.2f fps",
				clientID, len(d.frameBuffer), d.bufferSize,
				bufferStatus,
				float64(processDelay)/float64(time.Millisecond),
				fps)

		// 按自适应间隔从缓冲区获取帧并处理
		case <-outputTicker.C:
			// 检查是否达到自适应输出间隔
			if time.Since(lastOutputTime) < adaptiveInterval {
				continue
			}

			// 只有当没有正在处理的帧时才开始新的处理
			if !processingFrame {
				// 从缓冲区获取一帧
				select {
				case frame := <-d.frameBuffer:
					frameNum++
					processingFrame = true
					processStart := time.Now()
					lastOutputTime = processStart

					// 异步处理帧
					go func(f gocv.Mat, num int, startTime time.Time) {
						d.ProcessFrame(f, recvFrame, fps, num, byteTrack, trail, true)
						// 更新处理延迟
						processDelay = time.Since(startTime)
					}(frame, frameNum, processStart)
				default:
					// 缓冲区为空，等待下一个周期
				}
			}

		case buf := <-recvFrame:
			// 标记处理完成
			processingFrame = false

			if buf.Err != nil {
				log.Printf("客户端 #%d: 处理帧时发生错误: %v", clientID, buf.Err)
			} else {
				// 将图像写入响应
				w.Write([]byte("--frame\r\n"))
				w.Write([]byte("Content-Type: image/jpeg\r\n\r\n"))
				w.Write(buf.Buf.GetBytes())
				w.Write([]byte("\r\n"))

				// 刷新缓冲区
				flusher, ok := w.(http.Flusher)
				if ok {
					flusher.Flush()
				}
			}

			buf.Buf.Close()

			// 计算FPS
			frameCount++
			elapsed := time.Since(startTime).Seconds()

			if elapsed >= 1.0 {
				fps = float64(frameCount) / elapsed
				frameCount = 0
				startTime = time.Now()
			}
		}
	}
}

// handleClientDisconnect 处理客户端断开连接
func (d *Demo) handleClientDisconnect(clientID int) {
	// 加锁以保护共享资源
	d.rtspMutex.Lock()
	defer d.rtspMutex.Unlock()

	// 减少客户端计数
	d.clientCount--
	log.Printf("客户端 #%d 已断开连接，当前剩余客户端数量: %d", clientID, d.clientCount)

	// 如果没有更多客户端，关闭RTSP流
	if d.clientCount <= 0 {
		if d.rtspRunning && d.rtspExitCh != nil {
			log.Printf("最后一个客户端已断开连接，关闭RTSP流")
			// 发送信号关闭RTSP流
			d.rtspExitCh <- struct{}{}
			// 重置状态
			d.rtspRunning = false
			// 重置客户端计数（以防万一）
			d.clientCount = 0
		}
	}
}

// startRTSPStream 启动RTSP流并将帧复制到通道。该函数应从goroutine调用，因为它是阻塞的
func (d *Demo) startRTSPStream(framesCh chan gocv.Mat, exitCh chan struct{}) {
	var err error
	var rtspStream *gocv.VideoCapture

	rtspURL := d.rtspSrc.GetRTSPURL()
	log.Printf("正在连接RTSP流: %s\n", rtspURL)

	// 连接重试计数器
	retryCount := 0
	maxRetries := 10
	connected := false

	// 连接RTSP流，带重试
	for retryCount < maxRetries && !connected {
		rtspStream, err = gocv.OpenVideoCapture(rtspURL)
		if err != nil {
			retryCount++
			log.Printf("连接RTSP流失败 (尝试 %d/%d): %v", retryCount, maxRetries, err)
			time.Sleep(2 * time.Second)
		} else {
			connected = true
		}
	}

	if !connected {
		log.Printf("无法连接到RTSP流，达到最大重试次数: %d", maxRetries)
		return
	}

	defer rtspStream.Close()

	// 获取视频流的实际宽高
	width := int(rtspStream.Get(gocv.VideoCaptureFrameWidth))
	height := int(rtspStream.Get(gocv.VideoCaptureFrameHeight))
	log.Printf("RTSP流连接成功，分辨率: %dx%d\n", width, height)

	// 更新resizer使用实际分辨率
	if width > 0 && height > 0 {
		rt := d.pool.Get()
		d.resizer = preprocess.NewResizer(width, height,
			int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
		d.pool.Return(rt)
	}

	rtspImg := gocv.NewMat()
	defer rtspImg.Close()

	// 用于控制读取速率的计时器
	readTicker := time.NewTicker(time.Second / time.Duration(FPS))
	defer readTicker.Stop()

	// 用于监控连续错误的计数器
	consecutiveErrors := 0
	maxConsecutiveErrors := 5

	// 用于记录缓冲区统计信息
	frameCount := 0
	droppedFrames := 0

	// 每30秒打印一次缓冲区统计信息
	statsTicker := time.NewTicker(30 * time.Second)
	defer statsTicker.Stop()

loop:
	for {
		select {
		case <-exitCh:
			log.Printf("关闭RTSP流")
			// 清理帧缓冲区中的所有Mat对象，防止内存泄漏
			d.frameBufferMutex.Lock()
			if d.frameBuffer != nil {
				// 清空缓冲区中的所有帧
				log.Printf("清理帧缓冲区中的 %d 个帧", len(d.frameBuffer))
				for len(d.frameBuffer) > 0 {
					select {
					case frame := <-d.frameBuffer:
						// 关闭Mat对象释放内存
						frame.Close()
					default:
						// 缓冲区已空
						break
					}
				}
			}
			d.frameBufferMutex.Unlock()
			break loop

		case <-statsTicker.C:
			// 打印缓冲区统计信息
			log.Printf("RTSP流统计: 总帧数=%d, 丢弃帧数=%d, 丢帧率=%.2f%%, 缓冲区使用=%d/%d",
				frameCount, droppedFrames,
				float64(droppedFrames)/float64(max(frameCount, 1))*100.0,
				len(d.frameBuffer), cap(d.frameBuffer))

		case <-readTicker.C:
			if ok := rtspStream.Read(&rtspImg); !ok {
				// 读取RTSP帧出错，计数连续错误
				consecutiveErrors++
				log.Printf("读取RTSP帧错误 (%d/%d)，准备重新连接...", consecutiveErrors, maxConsecutiveErrors)

				// 如果连续错误超过阈值，重新连接
				if consecutiveErrors >= maxConsecutiveErrors {
					log.Printf("连续错误达到阈值，重新连接RTSP流")
					rtspStream.Close()
					time.Sleep(2 * time.Second)

					// 重置连接
					retryCount = 0
					connected = false

					// 尝试重新连接
					for retryCount < maxRetries && !connected {
						rtspStream, err = gocv.OpenVideoCapture(rtspURL)
						if err != nil {
							retryCount++
							log.Printf("重新连接RTSP流失败 (尝试 %d/%d): %v", retryCount, maxRetries, err)
							time.Sleep(3 * time.Second)
						} else {
							connected = true
							consecutiveErrors = 0 // 重置错误计数
							log.Printf("RTSP流重新连接成功")
						}
					}

					if !connected {
						log.Printf("无法重新连接到RTSP流，将在10秒后重试")
						time.Sleep(10 * time.Second)
					}
				}
				continue
			}

			// 成功读取帧，重置错误计数
			consecutiveErrors = 0

			if rtspImg.Empty() {
				continue
			}

			// 更新总帧数计数
			frameCount++

			// 发送帧到缓冲区，复制以避免竞态条件
			frameCopy := rtspImg.Clone()

			// 尝试将帧放入缓冲区，如果缓冲区已满则丢弃最旧的帧
			select {
			case d.frameBuffer <- frameCopy:
				// 成功添加到缓冲区
			default:
				// 缓冲区已满，丢弃当前帧
				frameCopy.Close()
				droppedFrames++
			}
		}
	}
}

// max 返回两个整数中的较大值
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ProcessFrame 从视频中获取图像并对其运行推理/对象检测，
// 注释图像并将结果编码为JPG文件返回
func (d *Demo) ProcessFrame(img gocv.Mat, retChan chan<- ResultFrame,
	fps float64, frameNum int, byteTrack *tracker.BYTETracker,
	trail *tracker.Trail, closeImg bool) {

	timing := &Timing{
		ProcessStart: time.Now(),
	}

	// 直接使用输入图像而不是复制，减少内存使用
	resImg := img
	if !closeImg {
		// 如果不能关闭原图像，则需要复制
		resImgCopy := gocv.NewMat()
		img.CopyTo(&resImgCopy)
		resImg = resImgCopy
		defer resImgCopy.Close()
	}

	// 对帧运行对象检测
	detectObjs, err := d.DetectObjects(resImg, frameNum, timing)

	if err != nil {
		log.Printf("Error detecting objects: %v", err)
		if closeImg {
			img.Close()
		}
		retChan <- ResultFrame{Err: err}
		return
	}

	if detectObjs == nil {
		// 未检测到对象
		// 编码图像为JPEG格式，使用较低的质量提高性能
		buf, err := gocv.IMEncodeWithParams(".jpg", resImg, []int{gocv.IMWriteJpegQuality, 80})
		if closeImg {
			img.Close()
		}
		retChan <- ResultFrame{Buf: buf, Err: err}
		return
	}

	detectResults := detectObjs.GetDetectResults()

	// 根据useTracking标志决定是否使用ByteTrack进行对象跟踪
	timing.TrackerStart = time.Now()

	var trackObjs []*tracker.STrack
	if d.useTracking {
		// 使用ByteTrack跟踪检测到的对象
		trackObjs, err = byteTrack.Update(
			postprocess.DetectionsToObjects(detectResults),
		)

		// 将跟踪的对象添加到历史轨迹
		for _, trackObj := range trackObjs {
			trail.Add(trackObj)
		}
	} else {
		// 不使用跟踪，直接将检测结果转换为跟踪对象格式以便统一渲染
		trackObjs = d.DetectionsToTracks(detectResults)
	}

	timing.TrackerEnd = time.Now()

	// 分割掩码创建必须在对象跟踪之后完成，因为跟踪的对象可能与对象检测结果不同
	var segMask postprocess.SegMask
	var keyPoints [][]postprocess.KeyPoint

	if d.modelType == "v5seg" {
		segMask = d.process.(*postprocess.YOLOv5Seg).TrackMask(detectObjs,
			trackObjs, d.resizer)

	} else if d.modelType == "v8seg" {
		segMask = d.process.(*postprocess.YOLOv8Seg).TrackMask(detectObjs,
			trackObjs, d.resizer)

	} else if d.modelType == "v8pose" {
		keyPoints = d.process.(*postprocess.YOLOv8Pose).GetPoseEstimation(detectObjs)
	}

	timing.DetObjEnd = time.Now()

	// 注释图像
	d.AnnotateImg(resImg, detectResults, trackObjs, segMask, keyPoints,
		trail, fps, frameNum, timing)

	// 编码图像为JPEG格式，使用较低的质量提高性能
	buf, err := gocv.IMEncodeWithParams(".jpg", resImg, []int{gocv.IMWriteJpegQuality, 80})

	res := ResultFrame{
		Buf: buf,
		Err: err,
	}

	if closeImg {
		// 关闭复制的网络摄像头帧
		img.Close()
	}

	// 注意：DetectionResult 接口没有 Free 方法
	// 资源清理由 defer 语句和垃圾回收处理

	retChan <- res
}

// DetectionsToTracks 将检测结果转换为跟踪对象格式，用于不使用ByteTrack时
func (d *Demo) DetectionsToTracks(detectResults []postprocess.DetectResult) []*tracker.STrack {
	var tracks []*tracker.STrack

	for _, det := range detectResults {
		// 创建一个新的跟踪对象，使用检测索引作为ID
		// 将 postprocess.BoxRect 转换为 tracker.Rect
		rect := tracker.NewRect(float32(det.Box.Left), float32(det.Box.Top),
			float32(det.Box.Right-det.Box.Left), float32(det.Box.Bottom-det.Box.Top))

		// 使用正确的字段名：Probability 而不是 Score，Class 而不是 ClassID
		track := tracker.NewSTrack(rect, det.Probability, int64(det.Class), det.Class)
		tracks = append(tracks, track)
	}

	return tracks
}

// LimitResults 获取跟踪结果并剔除我们不想跟踪的结果
func (d *Demo) LimitResults(trackResults []*tracker.STrack) []*tracker.STrack {

	if len(d.limitObjs) == 0 {
		return trackResults
	}

	// 剔除我们不想跟踪的检测对象
	var newTrackResults []*tracker.STrack

	for _, tResult := range trackResults {

		// 排除不是给定类别/标签的检测对象
		if len(d.limitObjs) > 0 {
			if !containsStr(d.limitObjs, d.labels[tResult.GetLabel()]) {
				continue
			}
		}

		newTrackResults = append(newTrackResults, tResult)
	}

	return newTrackResults
}

// AnnotateImg 在给定的图像Mat上绘制检测框和处理统计信息
func (d *Demo) AnnotateImg(img gocv.Mat, detectResults []postprocess.DetectResult,
	trackResults []*tracker.STrack,
	segMask postprocess.SegMask, keyPoints [][]postprocess.KeyPoint,
	trail *tracker.Trail, fps float64,
	frameNum int, timing *Timing) {

	timing.RenderingStart = time.Now()

	// 剔除我们不想跟踪的类别的跟踪结果
	trackResults = d.LimitResults(trackResults)
	objCnt := len(trackResults)

	if d.modelType == "v5seg" || d.modelType == "v8seg" {

		if d.renderFormat == "mask" {
			render.TrackerMask(&img, segMask.Mask, trackResults, detectResults, 0.5)

			render.TrackerBoxes(&img, trackResults, d.labels,
				render.DefaultFont(), 1)
		} else {
			render.TrackerOutlines(&img, segMask.Mask, trackResults, detectResults,
				1000, d.labels, render.DefaultFont(), 2, 5)
		}

	} else if d.modelType == "v8pose" {

		render.PoseKeyPoints(&img, keyPoints, 2)

		render.TrackerBoxes(&img, trackResults, d.labels,
			render.DefaultFont(), 1)

	} else if d.modelType == "v8obb" {

		render.TrackerOrientedBoundingBoxes(&img, trackResults, detectResults,
			d.labels, render.DefaultFontAlign(render.Center), 1)

	} else {
		// 绘制检测框
		render.TrackerBoxes(&img, trackResults, d.labels,
			render.DefaultFont(), 1)
	}

	// 只有在启用跟踪时才绘制对象轨迹线
	if d.modelType != "v8pose" && d.useTracking {
		render.Trail(&img, trackResults, trail, render.DefaultTrailStyle())
	}

	timing.ProcessEnd = time.Now()

	// 计算处理延迟
	lag := time.Since(timing.ProcessStart).Milliseconds() - int64(FPS)

	// 清空背景视频
	rect := image.Rect(0, 0, img.Cols(), 36)
	gocv.Rectangle(&img, rect, clrBlack, -1) // -1 填充矩形

	// 在图像顶部添加FPS、对象计数和帧号
	gocv.PutTextWithParams(&img, fmt.Sprintf("Frame: %d, FPS: %.2f, Lag: %dms, Objects: %d", frameNum, fps, lag, objCnt),
		image.Pt(4, 14), gocv.FontHersheySimplex, 0.5, clrWhite, 1,
		gocv.LineAA, false)

	// 在图像顶部添加推理统计信息
	gocv.PutTextWithParams(&img, fmt.Sprintf("Inference: %.2fms, Post Processing: %.2fms, Tracking: %.2fms, Rendering: %.2fms, Total Time: %.2fms",
		float32(timing.DetObjInferenceEnd.Sub(timing.DetObjStart))/float32(time.Millisecond),
		float32(timing.DetObjEnd.Sub(timing.DetObjInferenceEnd))/float32(time.Millisecond),
		float32(timing.TrackerEnd.Sub(timing.TrackerStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.RenderingStart))/float32(time.Millisecond),
		float32(timing.ProcessEnd.Sub(timing.ProcessStart))/float32(time.Millisecond),
	),
		image.Pt(4, 30), gocv.FontHersheySimplex, 0.5, clrWhite, 1,
		gocv.LineAA, false)
}

// DetectObjects 获取原始视频帧并对其运行YOLO推理以检测对象
func (d *Demo) DetectObjects(img gocv.Mat, frameNum int,
	timing *Timing) (postprocess.DetectionResult, error) {

	timing.DetObjStart = time.Now()

	// 转换颜色空间并调整图像大小（直接在原图上操作以减少内存使用）
	rgbImg := gocv.NewMat()
	defer rgbImg.Close()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := gocv.NewMat()
	defer cropImg.Close()

	// 使用LetterBoxResize调整图像大小以适应模型输入
	d.resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	// 对图像文件执行推理
	rt := d.pool.Get()
	outputs, err := rt.Inference([]gocv.Mat{cropImg})
	d.pool.Return(rt)

	if err != nil {
		return nil, fmt.Errorf("Runtime inferencing failed with error: %w", err)
	}

	timing.DetObjInferenceEnd = time.Now()

	// 执行对象检测
	detectObjs := d.process.DetectObjects(outputs, d.resizer)

	// 在完成后处理后释放在C内存中分配的输出
	err = outputs.Free()

	return detectObjs, nil
}

func main() {
	// disable logging timestamps
	log.SetFlags(0)

	// read in cli flags
	modelFile := flag.String("m", "../data/yolov5s-640-640-rk3588.rknn", "RKNN compiled YOLO model file")
	modelType := flag.String("t", "v5", "Version of YOLO model [v5|v8|v10|v11|x|v5seg|v8seg|v8pose]")
	rtspURL := flag.String("u", "rtsp://example.com/stream", "RTSP URL to connect to")
	codec := flag.String("codec", "h264", "Video codec [h264|h265]")
	username := flag.String("user", "", "RTSP username if authentication is required")
	password := flag.String("pass", "", "RTSP password if authentication is required")
	labelFile := flag.String("l", "../data/coco_80_labels_list.txt", "Text file containing model labels")
	httpAddr := flag.String("a", "localhost:8080", "HTTP Address to run server on, format address:port")
	poolSize := flag.Int("s", 3, "Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3")
	limitLabels := flag.String("x", "", "Comma delimited list of labels (COCO) to restrict object tracking to")
	renderFormat := flag.String("r", "outline", "The rendering format used for instance segmentation [outline|mask]")
	useTracking := flag.Bool("track", false, "Enable ByteTrack object tracking (default: false, only detection)")
	// 添加缓冲区大小参数，默认为FPS*3（约3秒的视频）
	bufferSize := flag.Int("buffer", FPS*3, "Frame buffer size to reduce stuttering (default: 3 seconds of video)")

	flag.Parse()

	if *poolSize > 33 {
		log.Fatalf("RKNN runtime pool size (flag -s) is to large, a value of 3, 6, 9, or 12 works best")
	}

	// 验证codec参数
	*codec = strings.ToLower(*codec)
	if *codec != "h264" && *codec != "h265" {
		log.Fatalf("Codec must be either 'h264' or 'h265'")
	}

	// 创建RTSP源
	rtspSrc := &RTSPSource{
		URL:      *rtspURL,
		Codec:    *codec,
		Username: *username,
		Password: *password,
	}

	err := rknnlite.SetCPUAffinity(rknnlite.RK3588FastCores)

	if err != nil {
		log.Printf("Failed to set CPU Affinity: %v\n", err)
	}

	demo, err := NewDemo(rtspSrc, *modelFile, *labelFile, *poolSize,
		*modelType, *renderFormat, rknnlite.RK3588)

	// 设置是否使用对象跟踪
	demo.useTracking = *useTracking
	if demo.useTracking {
		log.Println("ByteTrack object tracking enabled")
	} else {
		log.Println("ByteTrack object tracking disabled, using detection only")
	}

	// 设置帧缓冲区大小（帧缓冲区将在第一个客户端连接时创建）
	demo.bufferSize = *bufferSize
	// 初始化RTSP流退出通道
	demo.rtspExitCh = make(chan struct{})
	// 初始化客户端计数和RTSP流状态
	demo.clientCount = 0
	demo.rtspRunning = false
	log.Printf("Frame buffer size: %d frames (%.1f seconds)", demo.bufferSize, float64(demo.bufferSize)/float64(FPS))

	if err != nil {
		log.Fatalf("Error creating demo: %v", err)
	}

	if *limitLabels != "" {
		demo.LimitObjects(*limitLabels)
	}

	// 注册HTTP处理函数
	http.HandleFunc("/stream", demo.Stream)

	// start http server
	log.Println(fmt.Sprintf("Open browser and view video at http://%s/stream",
		*httpAddr))
	log.Fatal(http.ListenAndServe(*httpAddr, nil))
}
