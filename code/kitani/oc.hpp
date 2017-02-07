/*
 *
 *  oc.hpp
 *  ACTIVITY_FORECASTING_DEMO (simple optimal control demo)
 *
 *  Created by Kris Kitani on 11/27/12.
 *  Copyright 2012 Carnegie Mellon University. All rights reserved.
 *
 *  Activity Forecasting.
 *  Kris M. Kitani, Brian D. Ziebart, Drew Bagnell and Martial Hebert. 
 *  European Conference on Computer Vision (ECCV 2012).
 *
 
 Copyright (c) 2014, Kris Kitani
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class OC{
	
public:
	
	OC(){};
	~OC(){};

	void loadTerminalPts	(string input_filename);
	void loadRewardWeights	(string input_filename);
	void loadFeatures		(string input_filename);
	void loadImage			(string input_filename);
	
	int  computeValueFunction	(string output_filename);
	void computePolicy			(string output_filename);	
	void computeForecastDist	(string output_filename);
	
	void colormap				(Mat src, Mat &dst);
	void colormap_CumilativeProb(Mat src, Mat &dst);
	
	Mat _w;					// weight vector
	vector <Mat> feat;		// feature maps
	Mat _R;					// reward function
	Mat _V;					// value function
	vector<Mat> _Pax;		// policy
	Mat _D;					// cumilative sum of forecasting distribution
	Mat _img;				// rectified color image
	
	Point _start;			// start of trajectory
	Point _end;				// end of trajectory
  Point _end2;
	
	bool VISUALIZE;
};

