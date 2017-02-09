/*
*
*  DEMO OF ACTIVITY FORECASTING
*
*
*  Activity Forecasting.
*  Kris M. Kitani, Brian D. Ziebart, Drew Bagnell and Martial Hebert. 
*  European Conference on Computer Vision (ECCV 2012).
*
*/

#include <iostream>
#include "oc.hpp"
using namespace std;

int main (int argc, char * const argv[]) {
    String folder = argv[1];

	string birdseye_image_jpg_path		= "kitani/" + folder + "/walk_birdseye.jpg";
	string terminal_pts_txt_path		= "kitani/" + folder + "/walk_terminal_pts.txt";
	string reward_weights_txt_path		= "kitani/" + folder + "/walk_reward_weights.txt";
	string features_xml_path			= "kitani/" + folder + "/walk_feature_maps.xml";
	
	string output_valuefunc_xml_path	= "kitani/" + folder + "/output/walk_valuefunction.xml";
	string output_policy_xml_path		= "kitani/" + folder + "/output/walk_policy.xml";
	string output_jpg_path				= "kitani/" + folder + "/output/walk_forecast.jpg";
	
	OC model;
	model.folder_path = folder;

	//model.VISUALIZE = true;

	model.loadTerminalPts	(terminal_pts_txt_path);
	model.loadRewardWeights	(reward_weights_txt_path);
	model.loadFeatures		(features_xml_path);
	model.loadImage			(birdseye_image_jpg_path);
	
	model.computeValueFunction	(output_valuefunc_xml_path); 
	model.computePolicy			(output_policy_xml_path);
	model.computeForecastDist	(output_jpg_path);
	
    return 0;
}
