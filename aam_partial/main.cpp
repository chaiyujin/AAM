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

void test();

int main() {

	test();

	return 0;
}

void test() {
	aam::Model model;
	//model.buildFromDir("./data_my");
	model.buildFromVideo("./video/test.avi");
}