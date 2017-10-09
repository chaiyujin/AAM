#pragma once

#include <core/aam.h>
#include <utils/string.h>

namespace aam {
	class PCA {
	private:
		int numComponents;
		MatrixX W;
		RowVectorX mean;
		RowVectorX eigenValues;
	public:
		PCA(): mean(1, 1), eigenValues(0, 0), W(0, 0) {}
		PCA(const MatrixX &samplesOnRows, float percentage);
		template <class T>
		T project(const T &matrix) {
			auto m = matrix.rowwise() - mean;
			return m * W;
		}
		template <class T>
		T projectBack(const T &projected) {
			auto m = projected * W.transpose();
			return m.rowwise() + mean;
		}
		int getNumComponents() const { return numComponents; }
		const RowVectorX &getEigenValues() const {
			return eigenValues;
		}
		std::string toString() {
			std::string ret = "";
			ret += String::numToBytes<int>(numComponents);
			ret += String::matToBytes<MatrixX>(W);
			ret += String::matToBytes<RowVectorX>(mean);
			ret += String::matToBytes<RowVectorX>(eigenValues);
			return ret;
		}
		void fromStringStream(std::stringstream &str) {
			String::numFromBytes<int>(str, numComponents);
			String::matFromBytes<MatrixX>(str, W);
			String::matFromBytes<RowVectorX>(str, mean);
			String::matFromBytes<RowVectorX>(str, eigenValues);
		}
	};
}
