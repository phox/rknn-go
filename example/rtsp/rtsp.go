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
}

// NewDemo returns and instance of Demo, a streaming HTTP server showing
// video with object detection
func NewDemo(rtspSrc *RTSPSource, modelFile, labelFile string, poolSize int,
	modelType string, renderFormat string, cores []rknnlite.CoreMask) (*Demo, error) {

	var err error

	d := &Demo{
		rtspSrc:   rtspSrc,
		limitObjs: make([]string, 0),
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
	log.Printf("New client connection established\n")

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

	// 创建通道以接收来自RTSP的帧
	rtspFrames := make(chan gocv.Mat, 8)
	closeRTSP := make(chan struct{})

	// 启动RTSP流读取
	go d.startRTSPStream(rtspFrames, closeRTSP)

	frameNum := 0

loop:
	for {
		select {
		case <-r.Context().Done():
			log.Printf("Client disconnected\n")
			closeRTSP <- struct{}{}
			break loop

		// 接收RTSP帧
		case frame := <-rtspFrames:
			frameNum++

			go d.ProcessFrame(frame, recvFrame, fps, frameNum,
				byteTrack, trail, true)

		case buf := <-recvFrame:
			if buf.Err != nil {
				log.Printf("Error occured during ProcessFrame: %v", buf.Err)
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

// startRTSPStream 启动RTSP流并将帧复制到通道。该函数应从goroutine调用，因为它是阻塞的
func (d *Demo) startRTSPStream(framesCh chan gocv.Mat, exitCh chan struct{}) {
	var err error
	var rtspStream *gocv.VideoCapture

	rtspURL := d.rtspSrc.GetRTSPURL()
	log.Printf("Connecting to RTSP stream: %s\n", rtspURL)

	rtspStream, err = gocv.OpenVideoCapture(rtspURL)

	if err != nil {
		log.Printf("Error opening RTSP stream: %v", err)
		return
	}

	defer rtspStream.Close()

	// 获取视频流的实际宽高
	width := int(rtspStream.Get(gocv.VideoCaptureFrameWidth))
	height := int(rtspStream.Get(gocv.VideoCaptureFrameHeight))
	log.Printf("RTSP stream connected, resolution: %dx%d\n", width, height)

	// 更新resizer使用实际分辨率
	if width > 0 && height > 0 {
		rt := d.pool.Get()
		d.resizer = preprocess.NewResizer(width, height,
			int(rt.InputAttrs()[0].Dims[1]), int(rt.InputAttrs()[0].Dims[2]))
		d.pool.Return(rt)
	}

	rtspImg := gocv.NewMat()
	defer rtspImg.Close()

loop:
	for {
		select {
		case <-exitCh:
			log.Printf("Closing RTSP stream")
			break loop

		default:
			if ok := rtspStream.Read(&rtspImg); !ok {
				// 读取RTSP帧出错，尝试重新连接
				log.Printf("Error reading RTSP frame, attempting to reconnect...")
				rtspStream.Close()
				time.Sleep(2 * time.Second)

				rtspStream, err = gocv.OpenVideoCapture(rtspURL)
				if err != nil {
					log.Printf("Failed to reconnect to RTSP stream: %v", err)
					time.Sleep(5 * time.Second)
				}
				continue
			}
			if rtspImg.Empty() {
				continue
			}

			// 发送帧到通道，复制以避免竞态条件
			frameCopy := rtspImg.Clone()
			framesCh <- frameCopy

			// 添加一个小延迟以避免过度消耗CPU
			time.Sleep(time.Millisecond * 10)
		}
	}
}

// ProcessFrame 从视频中获取图像并对其运行推理/对象检测，
// 注释图像并将结果编码为JPG文件返回
func (d *Demo) ProcessFrame(img gocv.Mat, retChan chan<- ResultFrame,
	fps float64, frameNum int, byteTrack *tracker.BYTETracker,
	trail *tracker.Trail, closeImg bool) {

	timing := &Timing{
		ProcessStart: time.Now(),
	}

	resImg := gocv.NewMat()
	defer resImg.Close()

	// 复制源图像
	img.CopyTo(&resImg)

	// 对帧运行对象检测
	detectObjs, err := d.DetectObjects(resImg, frameNum, timing)

	if err != nil {
		log.Printf("Error detecting objects: %v", err)
		retChan <- ResultFrame{Err: err}
		return
	}

	if detectObjs == nil {
		// 未检测到对象
		if closeImg {
			img.Close()
		}

		// 编码图像为JPEG格式
		buf, err := gocv.IMEncode(".jpg", resImg)
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

	// 编码图像为JPEG格式
	buf, err := gocv.IMEncode(".jpg", resImg)

	res := ResultFrame{
		Buf: buf,
		Err: err,
	}

	if closeImg {
		// 关闭复制的网络摄像头帧
		img.Close()
	}

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

	// 转换颜色空间并调整图像大小
	rgbImg := gocv.NewMat()
	defer rgbImg.Close()
	gocv.CvtColor(img, &rgbImg, gocv.ColorBGRToRGB)

	cropImg := rgbImg.Clone()
	defer cropImg.Close()

	d.resizer.LetterBoxResize(rgbImg, &cropImg, render.Black)

	// 对图像文件执行推理
	rt := d.pool.Get()
	outputs, err := rt.Inference([]gocv.Mat{cropImg})
	d.pool.Return(rt)

	if err != nil {
		return nil, fmt.Errorf("Runtime inferencing failed with error: %w", err)
	}

	timing.DetObjInferenceEnd = time.Now()

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
