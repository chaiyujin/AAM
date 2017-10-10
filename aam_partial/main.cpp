#include <Eigen/Core>
#include <utils/asf.h>
#include <geo/procrustes.h>
#include <geo/triangulation.h>
#include <geo/mesh.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <texture/texture.h>
#include <utils/pca.h>
#include <utils/model.h>
#include <utils/texfit.h>
#include <utils/string.h>
#include <utils/save.h>

void test();
void test_model(aam::TexFitModel &model, std::string videoPath);

int main() {

	test();

	return 0;
}

void test() {
	aam::TexFitModel model;
	model.buildFromVideo("../video/test.avi");
	aam::Saver::save("../model/fit.model", model);
	aam::Saver::load("../model/fit.model", model);
	system("pause");
	test_model(model, "../video/test.avi");


	/*aam::Model model;
	model.buildFromVideo("../video/test.avi");*/

}

void test_model(aam::TexFitModel &model, std::string videoPath) {

	aam::MatrixX rawShapeList;
	cv::VideoCapture videoCapture;
	std::cout << "Load video and lms from " << videoPath << std::endl;
	aam::LMSUtil::loadFromVideo(videoPath, rawShapeList, videoCapture);
	model.testFitTexture(rawShapeList);
	videoCapture.release();
}