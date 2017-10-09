#pragma once

#include <core/aam.h>
#include <utils/pca.h>
#include <utils/lms.h>
#include <utils/asf.h>
#include <geo/mesh.h>
#include <geo/procrustes.h>
#include <geo/triangulation.h>
#include <texture/texture.h>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace aam {
	// find a texture for query shape
	class TexFitModel {
	private:
		PCA *p_pcaShp;
		MatrixX projectedShape;
		MatrixX_<byte> textureList;
		std::vector<int> sampleIndex;
		Point centerPoint;
		float xScale, yScale;
		void normalizeShapes(MatrixX &shapeList);
		void reset() {
			if (p_pcaShp) { delete p_pcaShp; p_pcaShp = nullptr; }
			sampleIndex.clear();
			centerPoint = Point(0, 0);
			xScale = yScale = 1;
		}
		void sort(std::vector<int> &idx, int l, int r);
		void qsort(std::vector<int> &idx, int l, int r);
		bool LT(const RowVectorX &X, const RowVectorX &Y);
		bool EQ(const RowVectorX &X, const RowVectorX &Y);
	public:
		TexFitModel() : p_pcaShp(nullptr) {}
		TexFitModel(std::istream &str) : p_pcaShp(nullptr) {
			this->fromistream(str);
		}
		void buildFromVideo(std::string videoPath);
		RowVectorX_<byte> fitTexture(const RowVectorX &queryShape, bool isRawShape, bool needNormalized);
		void normalizeShape(RowVectorX &shape) {
			auto &row = shape;
			for (int j = 0; j < row.cols(); j += 2) {
				float x = row[j];
				float y = row[j + 1];
				x = (x - centerPoint.x) / xScale;
				y = (y - centerPoint.y) / yScale;
				row[j] = x;
				row[j + 1] = y;
			}
		}
		void scaleShape(RowVectorX &normShape) {
			for (int i = 0; i < normShape.size(); i += 2) {
				normShape[i] = normShape[i] * xScale + xScale + 5;
				normShape[i + 1] = normShape[i + 1] * yScale + yScale + 5;
			}
		}
		std::string toString() const;
		bool fromistream(std::istream &);	
		void getScale(float &x, float &y) const {
			x = xScale;
			y = yScale;
		}
		// test
		void testFitTexture(const MatrixX &rawShapeList, bool raw = true, bool norm = true) {
			std::cout << sampleIndex.size() << std::endl;
			for (int i = 0; i < sampleIndex.size(); ++i) {
				fitTexture(rawShapeList.row(i), raw, norm);
			}
		}
	};
}
