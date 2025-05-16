# HTTP Stream with RTSP Source and ByteTrack Object Tracking

这个示例演示了如何从RTSP视频流中获取视频帧，使用YOLO模型进行对象检测，并通过ByteTrack进行对象跟踪，然后通过HTTP将处理后的视频流式传输到浏览器。

支持的YOLO模型包括YOLOv5、YOLOv8、YOLOv10、YOLOv11和YOLOX，同时支持H264和H265编解码器的RTSP流。

## 功能特点

- 支持RTSP视频流作为输入源
- 支持H264和H265视频编解码器
- 支持RTSP认证（用户名和密码）
- 使用YOLO模型进行对象检测
- 可选择使用ByteTrack进行对象跟踪和ID分配
- 通过HTTP将处理后的视频流式传输到浏览器
- 支持多种YOLO模型：YOLOv5、YOLOv8、YOLOv10、YOLOv11、YOLOX
- 支持实例分割（YOLOv5-seg、YOLOv8-seg）
- 支持姿态估计（YOLOv8-pose）
- 支持定向边界框（YOLOv8-obb）

## 统计信息显示

在视频流的顶部，会显示以下统计信息：

| 字段 | 描述 |
|------|------|
| Frame | 当前帧号 |
| FPS | 流式传输的每秒帧数 |
| Lag | 负值表示推理和处理时间在30 FPS的间隔内完成。正值表示处理时间更长，从视频首次开始播放时存在播放延迟。 |
| Objects | 场景中检测到的对象数量 |
| Inference | RKNN后端执行YOLO推理所需的时间 |
| Post Processing | 后处理YOLO检测结果所需的时间 |
| Tracking | 执行ByteTrack对象跟踪所需的时间 |
| Rendering | 绘制对象和用这些统计数据注释图像所需的时间 |
| Total Time | 从接收视频帧到处理完成发送到浏览器的总时间 |

## 使用方法

### 准备工作

确保您已经下载了数据文件。如果尚未下载，请执行以下命令：

```
cd example/
git clone https://github.com/phox/rknn-go-data.git data
```

### 运行示例

启动RTSP流处理服务器：

```
cd example/rtsp
go run rtsp.go -u rtsp://your-rtsp-url -codec h264
```

然后在浏览器中打开 http://localhost:8080/stream 查看处理后的视频流。

### 命令行参数

```
使用方法：
  go run rtsp.go [选项]

选项：
  -m string
        RKNN编译的YOLO模型文件 (默认 "../data/yolov5s-640-640-rk3588.rknn")
  -t string
        YOLO模型版本 [v5|v8|v10|v11|x|v5seg|v8seg|v8pose|v8obb] (默认 "v5")
  -u string
        要连接的RTSP URL (默认 "rtsp://example.com/stream")
  -codec string
        视频编解码器 [h264|h265] (默认 "h264")
  -user string
        如果需要认证，提供RTSP用户名
  -pass string
        如果需要认证，提供RTSP密码
  -l string
        包含模型标签的文本文件 (默认 "../data/coco_80_labels_list.txt")
  -a string
        运行服务器的HTTP地址，格式为address:port (默认 "localhost:8080")
  -s int
        RKNN运行时池大小，选择1、2、3或3的倍数 (默认 3)
  -x string
        限制对象跟踪的标签（COCO）的逗号分隔列表
  -r string
        用于实例分割的渲染格式 [outline|mask] (默认 "outline")
  -track
        启用ByteTrack对象跟踪 (默认为false，仅进行对象检测)
```

### 示例用法

1. 使用默认YOLOv5模型连接到RTSP流：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream
```

2. 使用YOLOv8模型和H265编解码器：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream -t v8 -codec h265
```

3. 使用需要认证的RTSP流：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream -user username -pass password
```

4. 限制只跟踪特定对象（例如，只跟踪人）：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream -x person
```

5. 使用YOLOv8-pose进行姿态估计：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream -t v8pose -m ../data/yolov8n-pose-640-640-rk3588.rknn -l ../data/yolov8_pose_labels_list.txt
```

6. 启用ByteTrack对象跟踪（默认不启用）：

```
go run rtsp.go -u rtsp://your-camera-ip:554/stream -track
```

## 注意事项

- 确保您的RTSP URL是正确的，包括正确的协议、IP地址、端口和路径
- 如果使用需要认证的RTSP流，请提供正确的用户名和密码
- 对于实例分割模型（v5seg、v8seg），由于CPU处理要求较高，FPS会自动降低到10
- 确保您使用的模型文件与指定的模型类型匹配