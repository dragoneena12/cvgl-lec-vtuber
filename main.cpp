#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <ApplicationServices/ApplicationServices.h>

#define PI 3.141592
#define EYE_METHOD 1 //0:using facemark, 1:using cascade

#define OUT_VIDEO_FILE "output_test.avi"

cv::Mat original_image, effected, prepare, prepare_buf, s_roi, d_roi, face, output, vtuber;
cv::Point2f begin, end;
cv::Point2f le_begin, le_end;
cv::Point2f re_begin, re_end;
cv::Point2f m_begin, m_end;
int prepare_step = 0;
int eyeopen_L = 1, eyeopen_R = 1;
float mouseopenrate = 1.0;
float faceangle = 0;
int mabutacolor[3], mousecolor[3];
double scale = 2.0;
double eyeopenmaxl = 0, eyeopenmaxr = 0, mouseopenmax = 0;
int hoho = 0;
int overlay[3] = {150, 150, 255};
int resultcolor[3]; 
int rec_mode = 0;

void MouseEventHandler(int event, int x , int y , int flags, void *){
  static bool isBrushDown = false;
  cv::Point pt(x, y);

  bool isLButtonPressedBeforeEvent = (bool)(flags & CV_EVENT_FLAG_LBUTTON);
  if(event == CV_EVENT_LBUTTONUP){
    end = pt;
    cv::Rect rect(begin, end);
    cv::Point center;
    int radius;
    center.x = cv::saturate_cast<int>((begin.x + end.x) * 0.5);
    center.y = cv::saturate_cast<int>((begin.y + end.y) * 0.5);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(prepare_buf, rrect, cv::Scalar(255, 0, 0), 1, 8);
    cv::imshow("image", prepare_buf);
  }

  // The XOR below means, isLButtonPressedAfterEvent
  // is usualy equal to isLButtonPressedBeforeEvent,
  // but not equal if the event is mouse down or up.
  bool isLButtonPressedAfterEvent = isLButtonPressedBeforeEvent
    ^ ((event == CV_EVENT_LBUTTONDOWN) || (event == CV_EVENT_LBUTTONUP));
  if((isLButtonPressedAfterEvent)){
    isBrushDown = true;
  }else{
    isBrushDown = false;
  }

  if(event == CV_EVENT_LBUTTONDOWN) {
      begin = pt;
      prepare_buf = prepare.clone();
      cv::imshow("image", prepare);
  }
}

  //目の開き具合検出
float getRaitoOfEyeOpen_L(std::vector<cv::Point2f> &points){
  if (eyeopenmaxl < std::abs(points[43].y - points[47].y)) eyeopenmaxl = std::abs(points[43].y - points[47].y);
    //printf("%f\n", std::abs(points[43].y - points[47].y) / eyeopenmaxl);
    return std::abs(points[43].y - points[47].y) / eyeopenmaxl;    
}

float getRaitoOfEyeOpen_R(std::vector<cv::Point2f> &points){
  if (eyeopenmaxr < std::abs(points[38].y - points[40].y)) eyeopenmaxr = std::abs(points[38].y - points[40].y);
    return std::abs(points[38].y - points[40].y) / eyeopenmaxr;    
}

//口の開き具合検出
float getRaitoOfMouseOpen(std::vector<cv::Point2f> &points){
  if (mouseopenmax < std::abs(points[51].y - points[57].y)) mouseopenmax = std::abs(points[51].y - points[57].y);
  //printf("%f\n", std::abs(points[51].y - points[57].y) / mouseopenmax);
  return std::abs(points[51].y - points[57].y) / mouseopenmax;    
}

//顔の傾き検出
float getAngleOfFace(std::vector<cv::Point2f> &points){
  //printf("%f\n", std::atan(std::abs(points[0].y - points[16].y)/std::abs(points[0].x - points[16].x)) * 180 / PI);
  return std::atan((points[0].y - points[16].y)/(points[0].x - points[16].x) * 180 / PI);
}

//顔パーツ指定（ステップ毎）
void imgprepare(bool* loop_flag){
  prepare_step++;
  if(prepare_step == 1) {
    cv::Rect rect(begin, end);
    cv::Point center;
    center.x = cv::saturate_cast<int>((begin.x + end.x) * 0.5);
    center.y = cv::saturate_cast<int>((begin.y + end.y) * 0.5);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(prepare, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::ellipse(prepare_buf, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::imshow("image", prepare);
    le_begin = begin;
    le_end = end;
    printf("set right eye and press 'o'\n");
  }
  if(prepare_step == 2) {
    cv::Rect rect(begin, end);
    cv::Point center;
    center.x = cv::saturate_cast<int>((begin.x + end.x) * 0.5);
    center.y = cv::saturate_cast<int>((begin.y + end.y) * 0.5);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(prepare, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::ellipse(prepare_buf, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::imshow("image", prepare);
    re_begin = begin;
    re_end = end;
    printf("select mabuta color\n");
  }
  if(prepare_step == 3){
    for(int i = 0; i < 3; i++) {
        mabutacolor[i] = (original_image.at<cv::Vec3b>(begin)[i] + original_image.at<cv::Vec3b>(begin)[i]) / 2;
    }
    printf("set mouse and press 'o'\n");
  }
  if(prepare_step == 4) {
    cv::Rect rect(begin, end);
    cv::Point center;
    center.x = cv::saturate_cast<int>((begin.x + end.x) * 0.5);
    center.y = cv::saturate_cast<int>((begin.y + end.y) * 0.5);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(prepare, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::ellipse(prepare_buf, rrect, cv::Scalar(0, 0, 255), 1, 8);
    cv::imshow("image", prepare);
    m_begin = begin;
    m_end = end;
    printf("select mouse color\n");
  }
  if(prepare_step == 5){
    for(int i = 0; i < 3; i++) {
        mousecolor[i] = (original_image.at<cv::Vec3b>(begin)[i] + original_image.at<cv::Vec3b>(begin)[i]) / 2;
    }
    printf("preparation compreted!\n");
    *loop_flag = false;
  }
}

void operate_image(){
  //operate image;
  effected = original_image.clone();
  if(eyeopen_L == 0) {
    cv::Rect rect(le_begin, le_end);
    cv::Point center;
    center.x = cv::saturate_cast<int>((le_begin.x + le_end.x) * 0.5);
    center.y = cv::saturate_cast<int>((le_begin.y + le_end.y) * 0.5 + 2);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(effected, rrect, cv::Scalar(0, 0, 0), -1, CV_AA);

    center.x = cv::saturate_cast<int>((le_begin.x + le_end.x) * 0.5);
    center.y = cv::saturate_cast<int>((le_begin.y + le_end.y) * 0.5 - 1);
    rect += cv::Size(4, 0);
    cv::RotatedRect rrect2(center, rect.size(), 0);
    cv::ellipse(effected, rrect2, cv::Scalar(mabutacolor[0], mabutacolor[1], mabutacolor[2]), -1, CV_AA);
  }
  if(eyeopen_R == 0) {
    cv::Rect rect(re_begin, re_end);
    cv::Point center;
    center.x = cv::saturate_cast<int>((re_begin.x + re_end.x) * 0.5);
    center.y = cv::saturate_cast<int>((re_begin.y + re_end.y) * 0.5 + 2);
    cv::RotatedRect rrect(center, rect.size(), 0);
    cv::ellipse(effected, rrect, cv::Scalar(0, 0, 0), -1, CV_AA);

    center.x = cv::saturate_cast<int>((re_begin.x + re_end.x) * 0.5);
    center.y = cv::saturate_cast<int>((re_begin.y + re_end.y) * 0.5 - 1);
    rect += cv::Size(4, 0);
    cv::RotatedRect rrect2(center, rect.size(), 0);
    cv::ellipse(effected, rrect2, cv::Scalar(mabutacolor[0], mabutacolor[1], mabutacolor[2]), -1, CV_AA);
  }
  {//口の変形
  //初期化
  cv::Rect rect(m_begin, m_end);
  cv::Point center;
  center.x = cv::saturate_cast<int>((m_begin.x + m_end.x) * 0.5);
  center.y = cv::saturate_cast<int>((m_begin.y + m_end.y) * 0.5);
  cv::RotatedRect rrect(center, rect.size(), 0);
  cv::ellipse(effected, rrect, cv::Scalar(mousecolor[0], mousecolor[1], mousecolor[2]), -1, CV_AA);
  
  //描画
  cv::Rect area(m_begin, m_end);
  s_roi = original_image(area);
  cv::Point m_begin2, m_end2;
  m_begin2.x = m_begin.x; m_end2.x = m_end.x;
  m_begin2.y = cv::saturate_cast<int>(m_begin.y + std::abs(m_begin.y - m_end.y) * (1.0 - mouseopenrate) / 2);
  m_end2.y = cv::saturate_cast<int>(m_end.y - std::abs(m_begin.y - m_end.y) * (1.0 - mouseopenrate) / 2);
  if(m_end2.y == m_begin2.y) m_end2.y = cv::saturate_cast<int>(m_end2.y + 1);
  //printf("%d, %d, %d, %d\n", m_begin2.x, m_begin2.y, m_end2.x, m_end2.y);
  cv::Rect area2(m_begin2, m_end2);
  d_roi = effected(area2);
  cv::resize(s_roi, d_roi, d_roi.size(), 0, 0);
  }

  {//draw hoho
    for (int i = 0; i < hoho * 20; i++) {
      
      cv::circle(effected, cv::Point((le_begin.x + le_end.x) / 2, (m_begin.y + m_end.y) / 2) + cv::Point(0.3 * i * std::cos(i), 0.3 * i * std::sin(i)), 1, cv::Scalar(resultcolor[0], resultcolor[1], resultcolor[2]), -1);
      cv::circle(effected, cv::Point((re_begin.x + re_end.x) / 2, (m_begin.y + m_end.y) / 2) + cv::Point(0.3 * i * std::cos(i), 0.3 * i * std::sin(i)), 1, cv::Scalar(resultcolor[0], resultcolor[1], resultcolor[2]), -1);
    }
  }

  {//顔の回転
  cv::Point2f center = cv::Point2f(static_cast<float>(original_image.cols / 2),static_cast<float>(original_image.rows));
  double degree = faceangle;  // 回転角度
  double scale = 1.0;    // 拡大率
  cv::Mat affine;
  cv::getRotationMatrix2D(center, degree, scale).copyTo(affine);
  cv::warpAffine(effected, effected, affine, original_image.size(), cv::INTER_CUBIC, 0, cv::Scalar(255, 255, 255));
  }
}

int main(int argc, char** argv){
  // read image file
  char *filename = (argc >= 2) ? argv[1] : (char *)"sample.jpg";
  original_image = cv::imread(filename);
  if(original_image.empty()){
    printf("ERROR: image not found!\n");
    return 0;
  }

  prepare = original_image.clone();
  prepare_buf = original_image.clone();
  cv::imshow("image", prepare);
  cv::namedWindow("image", 1);

  // set callback function for mouse operations
  cv::setMouseCallback("image", MouseEventHandler, 0);

  printf("set left eye and press 'o'\n");

  bool loop_flag = true;
  while(loop_flag){
    char key =cv::waitKey(50);
    if(key == 27){
      loop_flag = false;
      exit(1);
    } else if (key == 'o') {
      imgprepare(&loop_flag);
    }
  }

  // load classifier
  std::string cascadeName = "/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"; //Haar-like
  cv::CascadeClassifier cascade;
  if(!cascade.load(cascadeName)){
    printf("ERROR: cascadeFile not found\n");
    return -1;
  }

  cv::CascadeClassifier eyecascade;
  if(EYE_METHOD == 1){
    std::string eyecascadeName = "/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml"; //Haar-like
    if(!eyecascade.load(eyecascadeName)){
      printf("ERROR: cascadeFile not found\n");
      return -1;
    }
  }
  
  // initialize VideoCapture
  cv::Mat frame;
  cv::VideoCapture cap;
  cap.open(0);
  cap >> frame;
  if(frame.empty()) exit(1);

  // prepare facemark
  cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
  facemark->loadModel("lbfmodel.yaml");
  
  // prepare window
  cv::namedWindow("result", 1);
  cv::namedWindow("camera", 1);

  
  cv::Mat gray, smallImg(cv::saturate_cast<int>(frame.rows/scale),
               cv::saturate_cast<int>(frame.cols/scale), CV_8UC1);

  for (int j = 0; j < 3; j++) {
    //if (mousecolor[j] < 128) resultcolor[j] = mousecolor[j] * overlay[j] / 255 * 2;
    //else resultcolor[j] = 2 * (mousecolor[j] + overlay[j] - mousecolor[j] * overlay[j] / 255) - 255;
    resultcolor[j] = mousecolor[j] * overlay[j] / 255;
  }



  /* ディスプレイのサイズ */
  size_t width = CGDisplayPixelsWide(CGMainDisplayID());
  size_t height = CGDisplayPixelsHigh(CGMainDisplayID());

  /* RGBA用とBGR用のIplImageの作成 */
  IplImage *iplimage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);
  IplImage *bgrimage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
  IplImage *resizeimage = cvCreateImage(cvSize(width/2, height/2), IPL_DEPTH_8U, 3);

  /* グラフィックコンテキストの作成 */
  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  CGContextRef contextRef = CGBitmapContextCreate(
  iplimage->imageData, width, height,
  iplimage->depth, iplimage->widthStep,
  colorSpace,	kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);

  // prepare VideoWriter Object
  cv::Mat copy_frame;
  cv::namedWindow("output", 1);
  cv::VideoWriter output_video;
  output_video.open(OUT_VIDEO_FILE, CV_FOURCC('M', 'J', 'P', 'G'), 20, cv::Size(width/2, height/2));

  loop_flag = true;
  while(loop_flag) {
    effected = original_image.clone();

    // input from camera and resize
    cap >> frame;
    if(frame.empty()) break;
    cv::cvtColor(frame, gray, CV_BGR2GRAY);
    cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);

    // face detection
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0, cv::Size(100, 100));
    for(int i = 0; i < faces.size(); i++){
      cv::Point center;
      int radius;
      center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width * 0.5) * scale);
      center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height * 0.5) * scale);
      radius = cv::saturate_cast<int>((faces[i].width + faces[i].height) * 0.25 * scale);
      cv::Rect roi_rect(center.x - radius, center.y - radius, radius * 2, radius * 2);
      cv::rectangle(frame, roi_rect, cv::Scalar(255), 2, 8);
    }

    std::vector<cv::Rect> eyes;
    if(EYE_METHOD == 1 && faces.size() > 0){
      // eye detection
      face = smallImg(faces[0]);
      eyecascade.detectMultiScale(face, eyes, 1.15, 3, 0, cv::Size(10, 10));
      for(int i = 0; i < eyes.size(); i++){
        cv::Point center;
        int radius;
        center.x = cv::saturate_cast<int>((faces[i].x * scale) + (eyes[i].x + eyes[i].width * 0.5) * scale);
        center.y = cv::saturate_cast<int>((faces[i].y * scale) + (eyes[i].y + eyes[i].height * 0.5) * scale);
        radius = cv::saturate_cast<int>((eyes[i].width + eyes[i].height) * 0.25 * scale);
        cv::Rect roi_rect(center.x - radius, center.y - radius, radius * 2, radius * 2);
        cv::rectangle(frame, roi_rect, cv::Scalar(0, 255, 0), 2, 8);
      }
    }

    // landmark detection
    std::vector< std::vector<cv::Point2f> > landmarks;
    bool success = facemark->fit(smallImg,faces,landmarks);
    
    if(success){
      for(int i = 0; i < 1; i++)
      {
          for(int j = 0; j < landmarks[i].size(); j++) {
            cv::Point center;
            int radius;
            center.x = cv::saturate_cast<int>(landmarks[i][j].x * scale);
            center.y = cv::saturate_cast<int>(landmarks[i][j].y * scale);
            cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);
          }
      }
      if(EYE_METHOD == 0){
        if(getRaitoOfEyeOpen_L(landmarks[0]) > 0.6) eyeopen_L = 1;
        else eyeopen_L = 0;
        if(getRaitoOfEyeOpen_R(landmarks[0]) > 0.6) eyeopen_R = 1;
        else eyeopen_R = 0;
      } else {
        if(eyes.size() > 0) {
          eyeopen_L = 1;
          eyeopen_R = 1;
        } else {
          eyeopen_L = 0;
          eyeopen_R = 0;
        }
      }
      mouseopenrate = (getRaitoOfMouseOpen(landmarks[0]) - 0.5) * 3;
      if (mouseopenrate < 0.1) mouseopenrate = 0.1;
      if (mouseopenrate > 1.0) mouseopenrate = 1.0;
      faceangle = getAngleOfFace(landmarks[0]);
    }

    operate_image();

    //record
    /* メインディスプレイからCGImageRefを取得 */
    CGImageRef imageRef = CGDisplayCreateImage(CGMainDisplayID());
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), imageRef);

    /* RGBAからBGRに変換して，リサイズ */
    cvCvtColor(iplimage, bgrimage, CV_RGBA2BGR);
    cvResize(bgrimage, resizeimage);
    /* outputに */
    output = cv::cvarrToMat(resizeimage);
    CGImageRelease(imageRef);

    cv::Rect vtuberarea(0, output.rows - (200 * effected.rows / effected.cols), 200, 200 * effected.rows / effected.cols);
    cv::resize(effected, vtuber, cv::Size(200, 200 * effected.rows / effected.cols));
    vtuber.copyTo(output(vtuberarea));

    if(rec_mode){
      output_video << output;
      output.copyTo(copy_frame);
      cv::Size s = output.size();
      cv::rectangle(copy_frame, cv::Point(0, 0), cv::Point(s.width - 1, s.height - 1), cv::Scalar(0, 0, 255), 4, 8, 0);
      cv::imshow("output", copy_frame);  
    }
    else{
      cv::imshow("output", output);  
    }

    cv::imshow("camera", frame);
    cv::imshow("result", effected);

    char key =cv::waitKey(50);
    if(key == 27){
      loop_flag = false;
    }
    if(key == 'h'){
      if(hoho == 0) hoho = 1;
      else hoho = 0;
    }
    if(key == '0'){
      eyeopenmaxl = 0;
      eyeopenmaxr = 0;
      mouseopenmax = 0;
    }
    if(key == 'r'){
      if(rec_mode == 0){
        rec_mode = 1;
      }else{
        rec_mode = 0;
      }
    }
  }
}
  
