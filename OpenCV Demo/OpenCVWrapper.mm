//
//  OpenCVWrapper.mm
//  OpenCV Demo
//
//  Created by Drew Nibeck on 7/24/17.
//  Copyright © 2017 loc. All rights reserved.
//

#import "OpenCVWrapper.h"
#import <opencv2/opencv.hpp>
#import <opencv2/line_descriptor.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace line_descriptor;

@interface OpenCVWrapper ()

// MARK: Private
@property Ptr<BRISK> brisk;
@property FlannBasedMatcher flann;

@property Ptr<LSDDetector> lsd;
@property Ptr<BinaryDescriptor> bDes;
@property Ptr<BinaryDescriptorMatcher> bDesMatcher;

@property Mat queryCV;
@property Mat trainCV;

@property Mat queryDescriptors;
@property Mat trainDescriptors;

@property vector<KeyPoint> queryKeypoints;
@property vector<KeyPoint> trainKeypoints;

@property vector<KeyLine> queryKeylines;
@property vector<KeyLine> trainKeylines;

@end

@implementation OpenCVWrapper

// MARK: Setup/reset
-(id)init {
    _brisk = BRISK::create();
    _flann = FlannBasedMatcher(new flann::LshIndexParams(20, 10, 2));
    
    _lsd = LSDDetector::createLSDDetector();
    _bDes = BinaryDescriptor::createBinaryDescriptor();
    _bDesMatcher = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    
    return self;
}

-(void)reset {
    _area = 0;
    _error = NULL;
    
    _queryCV = NULL;
    _trainCV = NULL;
    
    _queryDescriptors = NULL;
    _trainDescriptors = NULL;
    
    _queryKeypoints = vector<KeyPoint>();
    _trainKeypoints = vector<KeyPoint>();
    
    _queryKeylines = vector<KeyLine>();
    _trainKeylines = vector<KeyLine>();
}

// MARK: Image Conversion

// Converts an UIImage into an opencv image object (colored) ** taken from opencv wiki **
-(Mat)cvImageFromUIImage: (UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

// Converts an UIImage into an opencv image object (grayscale) ** taken from opencv wiki **
-(Mat)cvGrayImageFromUIImage:(UIImage *)image {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels (grayscale)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

// Converts an opencv image object (grayscale or colored) into an UIImage ** taken from opencv wiki **
-(UIImage *)UIImageFromCVImage:(Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

// MARK: OpenCV to Swift (KeyPoints)
-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeypointsForQuery: (UIImage *) query {
    _queryCV = [self cvImageFromUIImage:query];
    cvtColor(_queryCV, _queryCV, COLOR_RGB2BGR);
    
    _queryKeypoints = [self calculateKeypoints:_queryCV];
    _queryDescriptors = [self calculateDescriptors:_queryCV withKeypoints:_queryKeypoints];
    
    return [self createSwiftKeypoints:_queryKeypoints];
}

-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeypointsForTrain: (UIImage *) train {
    _trainCV = [self cvImageFromUIImage:train];
    cvtColor(_trainCV, _trainCV, COLOR_RGB2BGR);
    
    _trainKeypoints = [self calculateKeypoints:_trainCV];
    _trainDescriptors = [self calculateDescriptors:_trainCV withKeypoints:_trainKeypoints];
    
    return [self createSwiftKeypoints:_trainKeypoints];
}

// MARK: OpenCV to Swift (KeyLines)
-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeylinesForQuery: (UIImage *) query {
    _queryCV = [self cvImageFromUIImage:query];
    cvtColor(_queryCV, _queryCV, COLOR_RGB2BGR);
    
    _queryKeylines = [self calculateKeyLines:_queryCV];
    _queryDescriptors = [self calculateDescriptors:_queryCV withKeyLines:_queryKeylines];
    
    return [self createSwiftKeylines:_queryKeylines];
}

-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeylinesForTrain: (UIImage *) train {
    _trainCV = [self cvImageFromUIImage:train];
    cvtColor(_trainCV, _trainCV, COLOR_RGB2BGR);
    
    _trainKeylines = [self calculateKeyLines:_trainCV];
    _trainDescriptors = [self calculateDescriptors:_trainCV withKeyLines:_trainKeylines];
    
    return [self createSwiftKeylines:_trainKeylines];
}

// MARK: Swift to OpenCV (KeyPoints)
-(void)setQueryKeypointsFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeypoints {
    _queryKeypoints = [self createCVKeypointsFromSwift:fromSwiftKeypoints];
}

-(void)setTrainKeypoints: (NSMutableArray<NSDictionary*> *) fromSwiftKeypoints {
    _trainKeypoints = [self createCVKeypointsFromSwift:fromSwiftKeypoints];
}

// MARK: Swift to OpenCV (KeyLines)
-(void)setQueryKeylinesFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeylines {
    _queryKeylines = [self createCVKeylinesFromSwift:fromSwiftKeylines];
}

-(void)setTrainKeylinesFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeylines {
    _trainKeylines = [self createCVKeylinesFromSwift:fromSwiftKeylines];
}

// MARK: Internal
-(NSMutableArray<NSDictionary *> *)createSwiftKeypoints: (vector<KeyPoint>) withCVKeypoints {
    NSMutableArray<NSDictionary *>* tmp = [[NSMutableArray<NSDictionary *> alloc] init];
    
    for (int i = 0; i < withCVKeypoints.size(); i++) {
        KeyPoint kp = withCVKeypoints[i];
        
        NSNumber* x =           [[NSNumber alloc] initWithFloat:kp.pt.x];
        NSNumber* y =           [[NSNumber alloc] initWithFloat:kp.pt.y];
        NSNumber* size =        [[NSNumber alloc] initWithFloat:kp.size];
        NSNumber* angle =       [[NSNumber alloc] initWithFloat:kp.angle];
        NSNumber* response =    [[NSNumber alloc] initWithFloat:kp.response];
        NSNumber* octave =      [[NSNumber alloc] initWithInt:kp.octave];
        NSNumber* classID =     [[NSNumber alloc] initWithInt:kp.class_id];
        
        NSDictionary* kpDict = [[NSDictionary alloc] initWithObjectsAndKeys:
                                x, @"x",
                                y, @"y",
                                size, @"size",
                                angle, @"angle",
                                response, @"response",
                                octave, @"octave",
                                classID, @"classID",
                                nil];
        
        [tmp addObject:kpDict];
    }
    
    return tmp;
}

-(NSMutableArray<NSDictionary *> *)createSwiftKeylines: (vector<KeyLine>) withCVKeylines {
    NSMutableArray<NSDictionary *>* tmp = [[NSMutableArray<NSDictionary *> alloc] init];
    
    for (int i = 0; i < withCVKeylines.size(); i++) {
        KeyLine kl = withCVKeylines[i];
        
        NSNumber* angle =           [[NSNumber alloc] initWithFloat:kl.angle];
        NSNumber* size =            [[NSNumber alloc] initWithFloat:kl.size];
        NSNumber* response =        [[NSNumber alloc] initWithFloat:kl.response];
        NSNumber* startPointX =     [[NSNumber alloc] initWithFloat:kl.startPointX];
        NSNumber* startPointY =     [[NSNumber alloc] initWithFloat:kl.startPointY];
        NSNumber* endPointX =       [[NSNumber alloc] initWithFloat:kl.endPointX];
        NSNumber* endPointY =       [[NSNumber alloc] initWithFloat:kl.endPointY];
        NSNumber* sPointInOctaveX = [[NSNumber alloc] initWithFloat:kl.sPointInOctaveX];
        NSNumber* sPointInOctaveY = [[NSNumber alloc] initWithFloat:kl.sPointInOctaveY];
        NSNumber* ePointInOctaveX = [[NSNumber alloc] initWithFloat:kl.ePointInOctaveX];
        NSNumber* ePointInOctaveY = [[NSNumber alloc] initWithFloat:kl.ePointInOctaveY];
        NSNumber* lineLength =      [[NSNumber alloc] initWithFloat:kl.lineLength];
        NSNumber* class_id =        [[NSNumber alloc] initWithInt:kl.class_id];
        NSNumber* octave =          [[NSNumber alloc] initWithInt:kl.octave];
        NSNumber* ptX =             [[NSNumber alloc] initWithInt:kl.pt.x];
        NSNumber* ptY =             [[NSNumber alloc] initWithInt:kl.pt.y];
        NSNumber* numOfPixels =     [[NSNumber alloc] initWithInt:kl.numOfPixels];
        
        NSDictionary* dict = [[NSDictionary alloc] initWithObjectsAndKeys:
                              angle, @"angle",
                              size, @"size",
                              response, @"response",
                              startPointX, @"startPointX",
                              startPointY, @"startPointY",
                              endPointX, @"endPointX",
                              endPointY, @"endPointY",
                              sPointInOctaveX, @"sPointInOctaveX",
                              sPointInOctaveY, @"sPointInOctaveY",
                              ePointInOctaveX, @"ePointInOctaveX",
                              ePointInOctaveY, @"ePointInOctaveY",
                              lineLength, @"lineLength",
                              class_id, @"class_id",
                              octave, @"octave",
                              ptX, @"ptX",
                              ptY, @"ptY",
                              numOfPixels, @"numOfPixels",
                              nil];
        
        [tmp addObject:dict];
    }
    
    return tmp;
}

-(vector<KeyPoint>)createCVKeypointsFromSwift: (NSMutableArray<NSDictionary *> *) withSwiftKeypoints {
    vector<KeyPoint> tmp;
    
    for (int i = 0; i < withSwiftKeypoints.count; i++) {
        NSDictionary* dict = withSwiftKeypoints[i];
        
        NSNumber* xNum =        [dict valueForKey:@"x"];        float x =        [xNum floatValue];
        NSNumber* yNum =        [dict valueForKey:@"y"];        float y =        [yNum floatValue];
        NSNumber* sizeNum =     [dict valueForKey:@"size"];     float size =     [sizeNum floatValue];
        NSNumber* angleNum =    [dict valueForKey:@"angle"];    float angle =    [angleNum floatValue];
        NSNumber* responseNum = [dict valueForKey:@"response"]; float response = [responseNum floatValue];
        NSNumber* octaveNum =   [dict valueForKey:@"octave"];   int   octave =   [octaveNum intValue];
        NSNumber* classIDNum =  [dict valueForKey:@"classID"];  int   classID =  [classIDNum intValue];
        
        KeyPoint kp = KeyPoint(x, y, size, angle, response, octave, classID);
        tmp.push_back(kp);
    }
    
    return tmp;
}

-(vector<KeyLine>)createCVKeylinesFromSwift: (NSMutableArray<NSDictionary *> *) withSwiftKeylines {
    vector<KeyLine> tmp;
    
    for (int i = 0; i < withSwiftKeylines.count; i++) {
        NSDictionary* dict = withSwiftKeylines[i];
        
        NSNumber* angleD =           [dict valueForKey:@"angle"];           float angle =           [angleD floatValue];
        NSNumber* sizeD =            [dict valueForKey:@"size"];            float size =            [sizeD floatValue];
        NSNumber* responseD =        [dict valueForKey:@"response"];        float response =        [responseD floatValue];
        NSNumber* startPointXD =     [dict valueForKey:@"startPointX"];     float startPointX =     [startPointXD floatValue];
        NSNumber* startPointYD =     [dict valueForKey:@"startPointY"];     float startPointY =     [startPointYD floatValue];
        NSNumber* endPointXD =       [dict valueForKey:@"endPointX"];       float endPointX =       [endPointXD floatValue];
        NSNumber* endPointYD =       [dict valueForKey:@"endPointY"];       float endPointY =       [endPointYD floatValue];
        NSNumber* sPointInOctaveXD = [dict valueForKey:@"sPointInOctaveX"]; float sPointInOctaveX = [sPointInOctaveXD floatValue];
        NSNumber* sPointInOctaveYD = [dict valueForKey:@"sPointInOctaveY"]; float sPointInOctaveY = [sPointInOctaveYD floatValue];
        NSNumber* ePointInOctaveXD = [dict valueForKey:@"ePointInOctaveX"]; float ePointInOctaveX = [ePointInOctaveXD floatValue];
        NSNumber* ePointInOctaveYD = [dict valueForKey:@"ePointInOctaveY"]; float ePointInOcatveY = [ePointInOctaveYD floatValue];
        NSNumber* lineLengthD =      [dict valueForKey:@"lineLength"];      float lineLength =      [lineLengthD floatValue];
        NSNumber* class_idD =        [dict valueForKey:@"class_id"];        int   class_id =        [class_idD intValue];
        NSNumber* octaveD =          [dict valueForKey:@"octave"];          int   octave =          [octaveD intValue];
        NSNumber* ptXD =             [dict valueForKey:@"ptX"];             int   ptX =             [ptXD intValue];
        NSNumber* ptYD =             [dict valueForKey:@"ptY"];             int   ptY =             [ptYD intValue];
        NSNumber* numOfPixelsD =     [dict valueForKey:@"numOfPixels"];     int   numOfPixels =     [numOfPixelsD intValue];

        KeyLine kl = KeyLine();
        kl.angle = angle;  kl.size = size;  kl.response = response;  kl.startPointX = startPointX;  kl.startPointY = startPointY;
        kl.endPointX = endPointX;  kl.endPointY = endPointY;  kl.sPointInOctaveX = sPointInOctaveX; kl.sPointInOctaveY = sPointInOctaveY;
        kl.ePointInOctaveX = ePointInOctaveX;  kl.ePointInOctaveY = ePointInOcatveY;  kl.lineLength = lineLength;  kl.class_id = class_id;
        kl.octave = octave;  kl.pt = Point2f(ptX, ptY);  kl.numOfPixels = numOfPixels;
        
        tmp.push_back(kl);
    }
    
    return tmp;
}

-(vector<KeyPoint>)calculateKeypoints: (Mat) forImage {
    vector<KeyPoint> kp;
    _brisk->detect(forImage, kp);
    
    return kp;
}

-(vector<KeyLine>)calculateKeyLines: (Mat) forImage {
    vector<KeyLine> kl;
    Mat mask = Mat::ones(forImage.size(), CV_8UC1);
    _lsd->detect(forImage, kl, 2, 1, mask);
    
    return kl;
}

-(Mat)calculateDescriptors: (Mat) image withKeypoints: (vector<KeyPoint>) keypoints {
    Mat des;
    _brisk->compute(image, keypoints, des);
        
    return des;
}

-(Mat)calculateDescriptors: (Mat) image withKeyLines: (vector<KeyLine>) keylines {
    Mat des;
    _bDes->compute(image, keylines, des);
    
    return des;
}

-(vector<DMatch>)findKeypointMatches {
    vector<DMatch> matches;
    _flann.match(_queryDescriptors, _trainDescriptors, matches);
    
    return matches;
}

-(vector<DMatch>)findKeylineMatches {
    vector<DMatch> matches;
    _bDesMatcher->match(_queryDescriptors, _trainDescriptors, matches);
    
    return matches;
}

-(vector<DMatch>)filterResults: (vector<DMatch>) unfiltered {
    double max_dist = 0;
    double min_dist = 100;
    
    for (int i = 0; i < unfiltered.size(); i++) {
        double dist = unfiltered[i].distance;
        if (dist < min_dist )
            min_dist = dist;
        
        if (dist > max_dist)
            max_dist = dist;
    }
    
    vector<DMatch> good;
    for(int i = 0; i < unfiltered.size(); i++) {
        if(unfiltered[i].distance <= max(2 * min_dist, 0.02)) {
            good.push_back(unfiltered[i]);
        }
    }
    
    return good;
}

-(Mat)findHomographyForKeypoints: (vector<DMatch>) withMatches {
    vector<Point2f> queryGoodKP;
    vector<Point2f> trainGoodKP;
    for (int i = 0; i < withMatches.size(); i++) {
        queryGoodKP.push_back(_queryKeypoints[withMatches[i].queryIdx].pt);
        trainGoodKP.push_back(_trainKeypoints[withMatches[i].trainIdx].pt);
    }
    
    Mat hSet = findHomography(queryGoodKP, trainGoodKP, CV_RANSAC);
    
    return hSet;
}

-(Mat)findHomographyForKeylines: (vector<DMatch>) withMatches {
    vector<Point2f> queryGoodKP;
    vector<Point2f> trainGoodKP;
    for (int i = 0; i < withMatches.size(); i++) {
        queryGoodKP.push_back(_queryKeylines[withMatches[i].queryIdx].pt);
        trainGoodKP.push_back(_trainKeylines[withMatches[i].trainIdx].pt);
    }
    
    Mat hSet = findHomography(queryGoodKP, trainGoodKP, CV_RANSAC);
    
    return hSet;
}

-(vector<Point2f>)findTransform: (Mat) hSet {
    vector<Point2f> queryCorners(4);
    queryCorners[0] = cvPoint(0,0); queryCorners[1] = cvPoint(_queryCV.cols, 0);
    queryCorners[2] = cvPoint(_queryCV.cols, _queryCV.rows); queryCorners[3] = cvPoint(0, _queryCV.rows);
    
    vector<Point2f> trainCorners;
    
    perspectiveTransform(queryCorners, trainCorners, hSet);
    
    return trainCorners;
}

// Returns true if lines intersect, false otherwise
-(Boolean)intersect: (Point2f)line1Start line1End: (Point2f) line1End line2Start: (Point2f) line2Start line2End: (Point2f) line2End {
    Point2f line1Dir = line1End - line1Start;
    Point2f line2Dir = line2End - line2Start;
    
    float cross = fabs( (line1Dir.x * line2Dir.y) - (line1Dir.y * line2Dir.x) );    // If 0, or really close to 0, lines intersect
    
    if (cross < 1e-8f) {  /* Threshold value */
        return true;
    }
    
    return false;
}

-(Boolean)polygonTest: (vector<Point2f>) corners {
    /*
         Assertion: There are only 4 lines!
     
         Line 1 = corners[0] -> corners[1]
         Line 2 = corners[1] -> corners[2]
         Line 3 = corners[2] -> corners[3]
         Line 4 = corners[3] -> corners[0]
     */

    // Line 1
    bool line1IntersectsLine2 = [self intersect:corners[0] line1End:corners[1] line2Start:corners[1] line2End:corners[2]];
    bool line1IntersectsLine3 = [self intersect:corners[0] line1End:corners[1] line2Start:corners[2] line2End:corners[3]];
    bool line1IntersectsLine4 = [self intersect:corners[0] line1End:corners[1] line2Start:corners[3] line2End:corners[0]];
    
    // Line 2
    bool line2IntersectsLine3 = [self intersect:corners[1] line1End:corners[2] line2Start:corners[2] line2End:corners[3]];
    bool line2IntersectsLine4 = [self intersect:corners[1] line1End:corners[2] line2Start:corners[3] line2End:corners[0]];
    
    // Line 3
    bool line3IntersectsLine4 = [self intersect:corners[2] line1End:corners[3] line2Start:corners[3] line2End:corners[0]];
    
    if (line1IntersectsLine2 || line1IntersectsLine3 || line1IntersectsLine4 ||
        line2IntersectsLine3 || line2IntersectsLine4 ||
        line3IntersectsLine4) {
        
        return false;
    }
    
    return true;
}

-(Boolean)doTests: (vector<Point2f>) trainCorners {
    if (trainCorners.size() != 4) {         // Quadrilateral test
        _error = [[NSString alloc] initWithString: @"Contour is not a quadrilateral"];
        cout << "Error: " << _error.cString << endl;
        
        return false;
    }
    
    if (![self polygonTest:trainCorners]) { // Polygon test
        _error = [[NSString alloc] initWithString: @"Contour is not a polygon"];
        cout << "Error: " << _error.cString << endl;
        
        return false;
    }
    
    double area = contourArea(trainCorners);
    _area = area;
    cout << "Area: " << area << endl;
    
    if (area < 10) {                        // Area test
        _error = [[NSString alloc] initWithString: @"Contour area is too small"];
        cout << "Error: " << _error.cString << endl;
        
        return false;
    }
    
    return true;
}

// MARK: Operations
-(UIImage *)matchWithKeypoints {
    vector<DMatch> matches = [self findKeypointMatches];
    cout << "Total number of matches: " << matches.size() << endl;
    
    vector<DMatch> good = [self filterResults:matches];
    cout << "Good matches: " << good.size() << endl;
    
    Mat outCV;
    drawMatches(_queryCV, _queryKeypoints, _trainCV, _trainKeypoints, good, outCV, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    Mat hSet = [self findHomographyForKeypoints:good];
    
    if (*hSet.size.p == NULL) {
        _error = [[NSString alloc] initWithString: @"No matches found"];
        cout << "Error: " << _error << endl;
        
        cvtColor(outCV, outCV, COLOR_BGR2RGB);
        UIImage* processed = [self UIImageFromCVImage: outCV];
        
        return processed;
    }
    
    vector<Point2f> trainCorners = [self findTransform:hSet];
    if (![self doTests:trainCorners]) {
        cvtColor(outCV, outCV, COLOR_BGR2RGB);
        UIImage* processed = [self UIImageFromCVImage: outCV];
        
        return processed;
    }
    
    line(outCV, trainCorners[0] + Point2f(_queryCV.cols, 0), trainCorners[1] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[1] + Point2f(_queryCV.cols, 0), trainCorners[2] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[2] + Point2f(_queryCV.cols, 0), trainCorners[3] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[3] + Point2f(_queryCV.cols, 0), trainCorners[0] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    
    cvtColor(outCV, outCV, COLOR_BGR2RGB);
    
    UIImage* processed = [self UIImageFromCVImage: outCV];
    
    return processed;
}

// Line descriptor extraction
-(UIImage *)matchWithKeylines {
    vector<DMatch> matches = [self findKeylineMatches];
    
    for (size_t i = 0; i < _queryKeylines.size(); i++ ) {
        KeyLine kl = _queryKeylines[i];
        if(kl.octave == 0) {
            // Random color
            int R = (rand() % (int) (255 + 1)); // Between 1 and 255
            int G = (rand() % (int) (255 + 1));
            int B = (rand() % (int) (255 + 1));

            // Line = pt1 -> pt2
            Point2f pt1 = Point2f(kl.startPointX, kl.startPointY);
            Point2f pt2 = Point2f(kl.endPointX, kl.endPointY);

            // Draw the line on the image using the random color
            line(_queryCV, pt1, pt2, Scalar(B, G, R), 3);  // line(img, startPt, endPt, color, thickness)
        }
    }
    
    for (size_t i = 0; i < _trainKeylines.size(); i++ ) {
        KeyLine kl = _trainKeylines[i];
        if(kl.octave == 0) {
            int R = (rand() % (int) (255 + 1));
            int G = (rand() % (int) (255 + 1));
            int B = (rand() % (int) (255 + 1));

            Point2f pt1 = Point2f(kl.startPointX, kl.startPointY);
            Point2f pt2 = Point2f(kl.endPointX, kl.endPointY);

            line(_trainCV, pt1, pt2, Scalar(B, G, R), 3);
        }
    }
    
    int maxHeight = _queryCV.rows > _trainCV.rows ? _queryCV.rows : _trainCV.rows;
    
    Mat outCV(maxHeight, _queryCV.cols + _trainCV.cols, _queryCV.type());
    Mat queryBox(outCV, cv::Rect(0, 0, _queryCV.cols, maxHeight));
    Mat trainBox(outCV, cv::Rect(_queryCV.cols, 0, _trainCV.cols, maxHeight));
    
    _queryCV.copyTo(queryBox);
    _trainCV.copyTo(trainBox);
    
    Mat hSet = [self findHomographyForKeylines:matches];
    
    if (*hSet.size.p == NULL) {
        _error = [[NSString alloc] initWithString: @"No matches found"];
        cout << "Error: " << _error << endl;
        
        cvtColor(outCV, outCV, COLOR_BGR2RGB);
        UIImage* processed = [self UIImageFromCVImage: outCV];
        
        return processed;
    }
    
    vector<Point2f> trainCorners = [self findTransform:hSet];
    if (![self doTests:trainCorners]) {
        cvtColor(outCV, outCV, COLOR_BGR2RGB);
        UIImage* processed = [self UIImageFromCVImage: outCV];
        
        return processed;
    }
    
    line(outCV, trainCorners[0] + Point2f(_queryCV.cols, 0), trainCorners[1] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[1] + Point2f(_queryCV.cols, 0), trainCorners[2] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[2] + Point2f(_queryCV.cols, 0), trainCorners[3] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    line(outCV, trainCorners[3] + Point2f(_queryCV.cols, 0), trainCorners[0] + Point2f(_queryCV.cols, 0), Scalar(0, 255, 0), 4);
    
    cvtColor(outCV, outCV, COLOR_BGR2RGB);
    UIImage* output = [self UIImageFromCVImage:outCV];
    
    return output;
}

// MARK: Debug
-(UIImage *)showKeypoints: (UIImage*) forImage {
    Mat cvImage = [self cvImageFromUIImage:forImage];
    cvtColor(cvImage, cvImage, COLOR_RGB2BGR);
    
    vector<KeyPoint> kp = [self calculateKeypoints:cvImage];
    
    Mat outCV;
    drawKeypoints(cvImage, kp, outCV);
    cvtColor(outCV, outCV, COLOR_BGR2RGB);
    
    return [self UIImageFromCVImage:outCV];
}

// Unfiltered
-(UIImage *)showKeypointMatches: (UIImage *) queryImage trainImage: (UIImage *)trainImage {
    Mat queryLocal = [self cvImageFromUIImage:queryImage];
    cvtColor(queryLocal, queryLocal, COLOR_RGB2BGR);
    
    Mat trainLocal = [self cvImageFromUIImage:trainImage];
    cvtColor(trainLocal, trainLocal, COLOR_RGB2BGR);
    
    vector<KeyPoint> kp1, kp2;
    _brisk->detect(queryLocal, kp1);
    _brisk->detect(trainLocal, kp2);
    
    Mat des1, des2;
    _brisk->compute(queryLocal, kp1, des1);
    _brisk->compute(trainLocal, kp2, des2);
    
    vector<DMatch> matches;
    _flann.match(des1, des2, matches);
    
    Mat outCV;
    drawMatches(queryLocal, kp1, trainLocal, kp2, matches, outCV);
    cvtColor(outCV, outCV, COLOR_BGR2RGB);
    
    return [self UIImageFromCVImage:outCV];
}

-(UIImage *)showKeylines: (UIImage *) forImage {
    Mat image = [self cvImageFromUIImage:forImage];
    
    vector<KeyLine> keyLines = [self calculateKeyLines:image];
    
    for (size_t i = 0; i < keyLines.size(); i++ ) {
        KeyLine kl = keyLines[i];
        if(kl.octave == 0) {
            int R = (rand() % (int) (255 + 1));
            int G = (rand() % (int) (255 + 1));
            int B = (rand() % (int) (255 + 1));
            
            Point2f pt1 = Point2f(kl.startPointX, kl.startPointY);
            Point2f pt2 = Point2f(kl.endPointX, kl.endPointY);
            
            line(image, pt1, pt2, Scalar(B, G, R), 3);
        }
    }
    
    return [self UIImageFromCVImage:image];
}

-(UIImage *)showKeylineMatches: (UIImage *) queryImage trainImage: (UIImage *) trainImage {
    Mat queryImageCV = [self cvImageFromUIImage:queryImage];
    cvtColor(queryImageCV, queryImageCV, CV_RGB2BGR);
    
    Mat trainImageCV = [self cvImageFromUIImage:trainImage];
    cvtColor(queryImageCV, queryImageCV, CV_RGB2BGR);
    
    vector<KeyLine> queryKL = [self calculateKeyLines:queryImageCV];
    vector<KeyLine> trainKL = [self calculateKeyLines:trainImageCV];
    
    Mat queryDes = [self calculateDescriptors:queryImageCV withKeyLines:queryKL];
    Mat trainDes = [self calculateDescriptors:trainImageCV withKeyLines:trainKL];
    
    vector<DMatch> matches;
    _bDesMatcher->match(queryDes, trainDes, matches);
    
    Mat outCV;
    drawLineMatches(queryImageCV, queryKL, trainImageCV, trainKL, matches, outCV);
    cvtColor(outCV, outCV, CV_BGR2RGB);
    
    return [self UIImageFromCVImage:outCV];
}

@end
