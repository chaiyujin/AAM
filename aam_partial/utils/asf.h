#pragma once

#include <core/aam.h>
#include <string>
#include <vector>

namespace aam {
	class ASFUtil {
	public:
		static bool loadFromASF(std::string filePath, RowVectorX &coords, std::string &imgPath);
		static bool loadFromDir(std::string dirPath, MatrixX &coordsList, std::vector<std::string> &imgList, bool needScale=true);
	};
}