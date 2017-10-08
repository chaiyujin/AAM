#include "asf.h"
#include <fstream>
#include <opencv2/opencv.hpp>

namespace aam {

	bool ASFUtil::loadFromASF(std::string filePath, RowVectorX &coords, std::string &imgPath) {
		std::ifstream fin(filePath);
		if (fin.good()) {
			int landmarkCount = 0;
			std::string line;
			while (std::getline(fin, line)) {
				if (line.length() == 0 || line[0] == '#') continue;
				if (line.find(".bmp") != std::string::npos ||
					line.find(".jpg") != std::string::npos ||
					line.find(".png") != std::string::npos) {
					imgPath = line;
					continue;
				}
				if (line.length() < 10) {
					int nbPoints = atol(line.c_str());
					coords.resize(1, nbPoints * 2);
				}
				else {
					std::stringstream stream(line);
					std::string path, type;
					std::string xStr, yStr;
					std::string pointIdStr, conn1Str, conn2Str;
					stream >> path >> type >> xStr >> yStr
						>> pointIdStr >> conn1Str >> conn2Str;

					aam::Scalar x = (aam::Scalar)atof(xStr.c_str());
					aam::Scalar y = (aam::Scalar)atof(yStr.c_str());

					int id = atoi(pointIdStr.c_str());
					int c1 = atoi(conn1Str.c_str());
					int c2 = atoi(conn2Str.c_str());

					coords(0, landmarkCount * 2 + 0) = x;
					coords(0, landmarkCount * 2 + 1) = y;

					landmarkCount++;
				}
			}
			fin.close();
			return true;
		}
		else {
			std::cout << "Failed to open file " << filePath << std::endl;
			return false;
		}
	}


	bool ASFUtil::loadFromDir(std::string dirPath, MatrixX &coordsMatrix, std::vector<std::string> &imgList, bool needScale) {
		if (dirPath[dirPath.length() - 1] != '/') dirPath += "/";
		std::string pathList = dirPath + "Path.txt";

		std::vector<RowVectorX> coordsList;
		bool ret = true;
		std::ifstream fin(pathList);
		std::string line;
		while (std::getline(fin, line)) {
			if (line.length() == 0) break;
			RowVectorX vec;
			std::string img;
			ret &= loadFromASF(line, vec, img);
			coordsList.push_back(vec);
			imgList.push_back(dirPath + img);
		}
		coordsMatrix.resize(coordsList.size(), coordsList[0].cols());
		for (size_t i = 0; i < coordsList.size(); ++i) {
			if (needScale) {
				std::string imgPath = imgList[i];
				cv::Mat mat = cv::imread(imgPath);
				for (int k = 0; k < coordsList[i].size(); k += 2) {
					coordsList[i][k] *= mat.cols;
					coordsList[i][k + 1] *= mat.rows;
				}
			}
			coordsMatrix.row(i) = coordsList[i];
		}
		return ret;
	}
}