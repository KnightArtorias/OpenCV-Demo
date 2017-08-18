//
//  OpenCVWrapper.h
//  OpenCV Demo
//
//  Created by Drew Nibeck on 7/24/17.
//  Copyright © 2017 loc. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface OpenCVWrapper : UIViewController

// Exposed properties
@property int area;
@property NSString* error;

// Create Swift models from OpenCV objects
-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeypointsForQuery: (UIImage *) query;
-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeypointsForTrain: (UIImage *) train;

-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeylinesForQuery: (UIImage *) query;
-(NSMutableArray<NSDictionary*> *)detectAndComputeWithKeylinesForTrain: (UIImage *) train;

// Create OpenCV models from Swift objects
-(void)setQueryKeypointsFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeypoints;
-(void)setTrainKeypointsFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeypoints;

-(void)setQueryKeylinesFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeylines;
-(void)setTrainKeylinesFromSwift: (NSMutableArray<NSDictionary*> *) fromSwiftKeylines;

// Operations
-(UIImage *)matchWithKeypoints;
-(UIImage *)matchWithKeylines;
-(void)reset;

// Debug
-(UIImage *)showKeypoints: (UIImage*) forImage;
-(UIImage *)showKeypointMatches: (UIImage *) queryImage trainImage: (UIImage *)trainImage;

-(UIImage *)showKeylines: (UIImage *) forImage;
-(UIImage *)showKeylineMatches: (UIImage *) queryImage trainImage: (UIImage *) trainImage;

@end
