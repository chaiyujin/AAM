#include "lms.h"

namespace aam {

	bool LMSUtil::loadFromVideo(std::string videoPath, MatrixX &coordsList, cv::VideoCapture &cap, bool needScale) {
		std::string lmsPath = videoPath.substr(0, videoPath.find_last_of('.')) + ".lms";
		std::ifstream fin(lmsPath);
		cap.open(videoPath);
		if (!cap.isOpened() || !fin.is_open()) {
			return false;
		}
		// set to frame 0
		cap.set(CV_CAP_PROP_POS_FRAMES, 0);
		std::string line;
		cv::Mat image;
		std::vector<std::vector<float>> list;
		while (std::getline(fin, line)) {
			if (line.length() == 0) break;
			cap >> image;
			float hc = image.cols / 2.f;
			float hr = image.rows / 2.f;
			std::vector<float> row;
			std::stringstream stream(line);
			float x, y;
			while (stream >> x >> y) {
				if (needScale) {
					x = x * hc + hc;
					y = y * hr + hr;
				}
				row.push_back(x);
				row.push_back(y);
			}
			list.push_back(row);
		}
		// copy to matrix
		coordsList.resize(list.size(), list[0].size());
		for (int i = 0; i < coordsList.rows(); ++i) {
			for (int j = 0; j < coordsList.cols(); ++j) {
				coordsList(i, j) = list[i][j];
			}
		}

		cap.set(CV_CAP_PROP_POS_FRAMES, 0);
		fin.close();
		return true;
	}
}