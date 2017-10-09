
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
	class Model {
	private:
		PCA *p_pcaShp;
		PCA *p_pcaTex;
		PCA *p_pcaApp;
		bool m_isGood;
		Point centerPoint;
		float xScale, yScale;
		float textureScale;
		void normalizeShapes(MatrixX &shapeList);
		void build(MatrixX &alignedShapeList, MatrixX &textureList);
		void reset() {
			if (p_pcaShp) { delete p_pcaShp; p_pcaShp = nullptr; }
			if (p_pcaTex) { delete p_pcaTex; p_pcaTex = nullptr; }
			if (p_pcaApp) { delete p_pcaApp; p_pcaApp = nullptr; }
			m_isGood = false;
			centerPoint = Point(0, 0);
			xScale = yScale = 1;
			textureScale = 1;
		}
	public:
		Model() : m_isGood(false), p_pcaShp(nullptr), p_pcaTex(nullptr), p_pcaApp(nullptr) {}
		~Model() { delete p_pcaShp; delete p_pcaTex; delete p_pcaApp; }

		void buildFromDir(std::string dirPath);
		void buildFromVideo(std::string videoPath);
		void normalizeShape(RowVectorX &shape) {
			for (int i = 0; i < shape.size(); i += 2) {
				shape[i] = (shape[i] - centerPoint.x) / xScale;
				shape[i + 1] = (shape[i + 1] - centerPoint.y) / yScale;
			}
		}
		void scaleShape(RowVectorX &normShape) {
			for (int i = 0; i < normShape.size(); i += 2) {
				normShape[i] = normShape[i] * xScale + centerPoint.x;
				normShape[i + 1] = normShape[i + 1] * yScale + centerPoint.y;
			}
		}
		template <class T>
		T combine(const T &shape, const T &texture) const {
			T pShp = p_pcaShp->project(shape);
			T pTex = p_pcaTex->project(texture);
			T concat(pShp.rows(), pShp.cols() + pTex.cols());
			concat << pShp, pTex * textureScale;
			return p_pcaApp->project(concat);
		}

		template <class T>
		void seperate(const T &app, T &shape, T &texture) const {
			T bApp = p_pcaApp->projectBack(app);
			T pShp = bApp.leftCols(p_pcaShp->getNumComponents());
			T pTex = bApp.rightCols(p_pcaTex->getNumComponents()) / textureScale;
			shape = p_pcaShp->projectBack(pShp);
			texture = p_pcaTex->projectBack(pTex);
		}

		bool good() const { return m_isGood; }
	};
}