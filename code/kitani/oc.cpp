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

#include "oc.hpp"

using namespace std;

void OC::loadTerminalPts(string input_filename)
{
	cout << "\nLoadTerminalPts()\n";
	ifstream fs;
	fs.open(input_filename.c_str());
	if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	fs >> _start.x;		fs >> _start.y;
	fs >> _end.x;		fs >> _end.y;
	cout << "  start:" << _start << endl;
	cout << "  end:" << _end << endl;
	
}

void OC::loadRewardWeights	(string input_filename)
{
	cout << "\nLoadRewardWeights()\n";
	ifstream fs;
	fs.open(input_filename.c_str());
	if(!fs.is_open()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	float val;
	while(fs >> val)_w.push_back(val);
	cout << "  Number of weights loaded:" << _w.rows << endl;
}

void OC::loadFeatures	(string input_filename)
{
	cout << "\nLoadFeatures()\n";
	FileStorage fs;
	fs.open(input_filename.c_str(), FileStorage::READ);
	if(!fs.isOpened()){cout << "ERROR: Opening: " << input_filename << endl;exit(1);}
	
	for(int i=0;true;i++)
	{
		stringstream ss;
		ss << "feature_" << i;
		Mat tmp;
		fs[ss.str()] >> tmp;
		if(!tmp.data) break;
		feat.push_back(tmp+0.0);
	}
	cout << "  Number of feature maps loaded:" << feat.size() << endl;
}

void OC::loadImage	(string input_filename)
{
	_img = imread(input_filename);
	if(!_img.data){cout << "ERROR: Opening:" << input_filename << endl; exit(1);}
	resize(_img,_img,feat[0].size());
	
	if(VISUALIZE)
	{
		imshow("Bird's Eye Image", _img);
		waitKey(100);
	}
	
}

int OC::computeValueFunction	(string output_filename)
{
	//cout << "\nComputeValueFunction()\n";
	_R = Mat::zeros(feat[0].size(),CV_32FC1);
	for(int i=0;i<(int)_w.rows;i++) {
		_R += _w.at<float>(i,0) * feat[i];
	}
	if(VISUALIZE)
	{
		Mat dsp;
		colormap(_R,dsp);
		imshow("Reward Function",dsp);
		waitKey(100);
	}
	
	Mat V;
	_V	= Mat::ones(feat[0].size(),CV_32FC1) * -FLT_MAX;
	V	= _V.clone();
	
	int n=0;
	
	while(1)
	{
		Mat V_padded;
		Mat v = _V * 1.0;
		copyMakeBorder(v,V_padded,1,1,1,1,BORDER_CONSTANT,Scalar::all(-FLT_MAX));
		V_padded *= 1.0;
		
		for(int col=0;col<(V_padded.cols-2);col++)
		{
			for(int row=0;row<(V_padded.rows-2);row++)
			{
				Mat sub = V_padded(Rect(col,row,3,3));
				double minVal, maxVal;
				minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
				
				if(maxVal==-FLT_MAX) continue;			// entire region has no data
				
				// ===== SOFTMAX ===== //
				for(int y=0;y<3;y++)					// softmax over actions
				{
					for(int x=0;x<3;x++)				// softmax over actions
					{
						if(y==1 && x==1) continue;		// stopping prohibited
						
						float minv = MIN(_V.at<float>(row,col),sub.at<float>(y,x));
						float maxv = MAX(_V.at<float>(row,col),sub.at<float>(y,x));
						
						float softmax = maxv + log(1.0 + exp(minv-maxv));
						_V.at<float>(row,col) = softmax;
					}
				}
				_V.at<float>(row,col) += _R.at<float>(row,col);
			}
		}
		_V.at<float>(_end.y,_end.x) = 0.0;			// set goal value to 0
		
		
		// ==== CONVERGENCE CRITERIA ==== //
		Mat residual;
		double minVal, maxVal;
		absdiff(_V,V,residual);
		minMaxLoc(residual,&minVal,&maxVal,NULL,NULL);
		_V.copyTo(V);
		
		if(maxVal<.9) break;
		
		n++;
		if(n>2000){cout << "ERROR: Max number of iterations." << endl;exit(1);}
	}
	
	cout << "  Converged in " << n << " steps.\n";
	
	FileStorage fs;
	fs.open(output_filename,FileStorage::WRITE);
	if(fs.isOpened()) cout << "  Writing:" << output_filename << endl;
	fs << "ValueFunction" << _V;
	fs.release();
	
	return 0;
}

void OC::computePolicy			(string output_filename)
{
	cout << "\nComputePolicy()\n";
	int na = 9;				// number of actions (3x3)
	
	for(int a=0;a<na;a++) _Pax.push_back(Mat::zeros(_V.size(),CV_32FC1));	// allocate memory
	
	double minVal, maxVal;
	Mat V_padded;
	copyMakeBorder(_V,V_padded,1,1,1,1,BORDER_CONSTANT,Scalar(-INFINITY));
	
	for(int col=0;col<V_padded.cols-2;col++)
	{
		for(int row=0;row<V_padded.rows-2;row++)
		{
			Rect r(col,row,3,3);
			Mat sub = V_padded(r);
			minMaxLoc(sub,&minVal,&maxVal,NULL,NULL);
			Mat p = sub - maxVal;				// log rescaling
			exp(p,p);							// Z(x,a) - probability space
			p.at<float>(1,1) = 0;				// zero out center
			Scalar su = sum(p);					// sum (denominator)
			if(su.val[0]>0) p /= su.val[0];		// normalize (compute policy(x|a))
			else p = 1.0/(na-1.0);				// uniform distribution
			p = p.reshape(1,1);					// vectorize
			for(int a=0;a<na;a++) _Pax[a].at<float>(row,col) = p.at<float>(0,a); // update policy
		}
		
	}
	
	FileStorage fs;
	fs.open(output_filename,FileStorage::WRITE);
	cout << "  Writing: " << output_filename << endl;
	for(int a=0;a<na;a++)
	{
		stringstream ss;
		ss << "action_" << a;
		fs << ss.str() << _Pax[a];
	}
	fs.release();
}	

void OC::computeForecastDist	(string output_filename)
{
	cout << "\nComputeForecastDist()\n";
	
	_D = Mat::zeros(_V.size(),CV_32FC1);
	
	Mat N[2];
	N[0] = _D.clone();
	N[1] = _D.clone();
	N[0].at<float>(_start.y,_start.x) = 1.0;					// initialize start
	
	Mat dsp;
	
	int n=0;
	while(1)
	{
		N[1] *= 0.0;
		for(int col=0;col<N[0].cols;col++)
		{
			for(int row=0;row<N[0].rows;row++)
			{
				if(row==_end.y && col == _end.x) continue;		// absorbsion state
				
				if(N[0].at<float>(row,col) > (FLT_MIN))
				{
					int col_1 = N[1].cols-1;
					int row_1 = N[1].rows-1;
					
					if(col>0	 && row>0	 )	N[1].at<float>(row-1,col-1) += N[0].at<float>(row,col) * _Pax[0].at<float>(row,col);	// NW
					if(				row>0	 )	N[1].at<float>(row-1,col-0) += N[0].at<float>(row,col) * _Pax[1].at<float>(row,col);	// N
					if(col<col_1 && row>0	 )	N[1].at<float>(row-1,col+1) += N[0].at<float>(row,col) * _Pax[2].at<float>(row,col);	// NE
					if(col>0				 )  N[1].at<float>(row-0,col-1) += N[0].at<float>(row,col) * _Pax[3].at<float>(row,col);	// W
					if(col<col_1             )	N[1].at<float>(row-0,col+1) += N[0].at<float>(row,col) * _Pax[5].at<float>(row,col);	// E
					if(col>0	 && row<row_1)	N[1].at<float>(row+1,col-1) += N[0].at<float>(row,col) * _Pax[6].at<float>(row,col);	// SW
					if(			    row<row_1)	N[1].at<float>(row+1,col-0) += N[0].at<float>(row,col) * _Pax[7].at<float>(row,col);	// S
					if(col<col_1 && row<row_1)	N[1].at<float>(row+1,col+1) += N[0].at<float>(row,col) * _Pax[8].at<float>(row,col);	// SE
					
				}
			}
		}
		N[1].at<float>(_end.y,_end.x) = 0.0;				// absorption state
		
		_D += N[1];
		FileStorage fs;
		fs.open("kitani/oc_demo/frames/frame"+ to_string(n) + ".xml",FileStorage::WRITE);
		stringstream ss;
		ss << "img";
		fs << ss.str() << _D;
		fs.release();
		cout << "Wrote frame " << n << "\n";
		
		if(VISUALIZE)
		{
			
			colormap_CumilativeProb(_D,dsp);
			_img.copyTo(dsp,dsp<1);
			addWeighted(dsp,0.5,_img,0.5,0,dsp);
			imshow("Forecast Distribution",dsp);
			waitKey(1);
		}
		
		swap(N[0],N[1]);
		
		if(n++>300) break;									// trajectory dependent
	}
	colormap_CumilativeProb(_D,dsp);
	cout << "  Writing:" << output_filename << endl;
	imwrite(output_filename,dsp);
	cout << "\nPress any key to finish.\n";
	//waitKey(0);
	
	
}

void OC::colormap(Mat _src, Mat &dst)
{
	if(_src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	double minVal,maxVal;
	Mat src;
	_src.copyTo(src);
	Mat isInf;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	compare(src,-FLT_MAX,isInf,CMP_GT);
	threshold(src,src,-FLT_MAX,0,THRESH_TOZERO);
	minMaxLoc(src,&minVal,NULL,NULL,NULL);
	Mat im = (src-minVal)/(maxVal-minVal) * 255.0;
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);
	I3[0] = U8 * 0.85;
	I3[1] = isInf;
	I3[2] = isInf;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
}


void OC::colormap_CumilativeProb(Mat src, Mat &dst)
{
	if(src.type()!=CV_32FC1) cout << "ERROR(jetmap): must be single channel float\n";
	
	Mat im;
	src.copyTo(im);
	
	double minVal = 1e-4;
	double maxVal = 0.2;
	threshold(im,im,minVal,0,THRESH_TOZERO);
	
	im = (im-minVal)/(maxVal-minVal)*255.0;
	
	Mat U8,I3[3],hsv;
	im.convertTo(U8,CV_8UC1,1.0,0);	
	I3[0] = U8*1.0;									// Hue
	
	Mat pU;
	U8.convertTo(pU,CV_64F,-1.0/255.0,1.0);
	pow(pU,0.5,pU);									
	pU.convertTo(U8,CV_8UC1,255.0,0);					
	I3[1] = U8*1.0;									// Saturation
	
	Mat isNonZero;
	compare(im,0,isNonZero,CMP_GT);
	I3[2] = isNonZero;								// Value
	
	merge(I3,3,hsv);
	circle(hsv, _start, 4, Scalar( 255, 255, 255 ), 10);
	circle(hsv, _end, 4, Scalar( 255, 255, 255 ), 10);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);				// Convert to RGB
}
