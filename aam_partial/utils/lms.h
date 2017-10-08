#pragma once

#include <core/aam.h>

namespace aam {
	class LMSUtil {
	public:
		static bool loadFromVideo(std::string videoPath, MatrixX &coordsList, cv::VideoCapture &cap, bool needScale = true);
	};
}
