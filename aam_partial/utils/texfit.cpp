#include "texfit.h"
#include "print.h"

namespace aam {

	void TexFitModel::normalizeShapes(MatrixX &shapeList) {
		// center
		centerPoint = Point(0, 0);
		const auto &meanShape = Procrustes::getMeanShape();
		float min_x = meanShape[0], max_x = meanShape[0];
		float min_y = meanShape[1], max_y = meanShape[1];
		for (int j = 0; j < meanShape.size(); j += 2) {
			float x = meanShape[j];
			float y = meanShape[j + 1];
			if (x < min_x) min_x = x;
			if (x > max_x) max_x = x;
			if (y < min_y) min_y = y;
			if (y > max_y) max_y = y;
		}
		centerPoint.x = (min_x + max_x) / 2.;
		centerPoint.y = (min_y + max_y) / 2.;
		// scale
		xScale = 1;
		yScale = 1;
		for (int i = 0; i < shapeList.rows(); ++i) {
			auto &row = shapeList.row(i);
			for (int j = 0; j < row.cols(); j += 2) {
				float x = row[j];
				float y = row[j + 1];
				x = x - centerPoint.x;
				y = y - centerPoint.y;
				if (std::abs(x) > xScale) xScale = std::abs(x);
				if (std::abs(y) > yScale) yScale = std::abs(y);
			}
		}
		for (int i = 0; i < shapeList.rows(); ++i) {
			auto &row = shapeList.row(i);
			for (int j = 0; j < row.cols(); j += 2) {
				float x = row[j];
				float y = row[j + 1];
				x = (x - centerPoint.x) / xScale;
				y = (y - centerPoint.y) / yScale;
				row[j] = x;
				row[j + 1] = y;
			}
		}
	}

	void TexFitModel::buildFromVideo(std::string videoPath) {
		this->reset();
		MatrixX rawShapeList;
		cv::VideoCapture videoCapture;
		std::cout << "Load video and lms from " << videoPath << std::endl;
		LMSUtil::loadFromVideo(videoPath, rawShapeList, videoCapture);
		std::cout << "Align and normalize shapes\n";
		auto alignedShapeList = Procrustes::alignShapes(rawShapeList, 10);
		this->normalizeShapes(alignedShapeList);
		std::cout << "Triangulation\n";
		auto triangleIndexes = Triangulation::delaunayTriangulation(Procrustes::getMeanShape());
		auto meanMesh = Mesh(Procrustes::getMeanShape());
		auto texCoords = Texture::texCoordsInMeanMesh(meanMesh);
		std::cout << "Collect texture\n";
		textureList = Texture::collectTextures(rawShapeList, videoCapture);

		videoCapture.release();

		// pca on aligned shape
		std::cout << "PCA on shape\n";
		p_pcaShp = new PCA(alignedShapeList, 0.50);
		std::cout << "Shape has components: " << p_pcaShp->getNumComponents() << std::endl;
		projectedShape = p_pcaShp->project(alignedShapeList);
		//// sort samples
		for (int i = 0; i < projectedShape.rows(); ++i) {
			sampleIndex.push_back(i);
		}
		this->qsort(sampleIndex, 0, sampleIndex.size() - 1);

		//testFitTexture(alignedShapeList);
		//testFitTexture(rawShapeList);
	}

	void TexFitModel::sort(std::vector<int> &idx, int l, int r) {
		for (int i = l; i <= r; ++i) {
			for (int j = i + 1; j <= r; ++j) {
				if (LT(projectedShape.row(idx[j]), projectedShape.row(idx[i]))) {
					int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
				}
			}
		}
	}

	void TexFitModel::qsort(std::vector<int> &idx, int l, int r) {
		int i = l, j = r;
		int m = (i + j) / 2;
		RowVectorX key(1, projectedShape.row(idx[m]).cols());
		key << projectedShape.row(idx[m]);
		while (i <= j) {
			while (LT(projectedShape.row(idx[i]), key) && i < r) ++i;
			while (LT(key, projectedShape.row(idx[j])) && l < j) --j;
			if (i <= j) {
				int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
				++i; --j;
			}
		}
		if (i <= r) qsort(idx, i, r);
		if (l <= j) qsort(idx, l, j);
	}

	bool TexFitModel::LT(const RowVectorX &X, const RowVectorX &Y) {
		for (int i = 0; i < X.size(); ++i) {
			if (X[i] > Y[i]) return false;
			if (X[i] < Y[i]) return true;
		}
		return false;
	}

	bool TexFitModel::EQ(const RowVectorX &X, const RowVectorX &Y) {
		for (int i = 0; i < X.size(); ++i) {
			if (!(std::abs(X[i] - Y[i]) < 1e-6)) return false;
		}
		return true;
	}

	RowVectorX TexFitModel::fitTexture(const RowVectorX &queryShape, bool isRawShape, bool needNormalized) {
		// project queryShape
		RowVectorX shape = queryShape;
		if (isRawShape) Procrustes::procrustes(Procrustes::getMeanShape(), shape);
		if (isRawShape || needNormalized) normalizeShape(shape);
		auto query = p_pcaShp->project(shape);
		int l = 0, r = projectedShape.rows() - 1;
		int idx = -1;
		while (l <= r) {
			int m = (l + r) / 2;
			const auto & key = projectedShape.row(sampleIndex[m]);
			idx = m;
			if (EQ(query, key)) {
				break;
			}
			else if (LT(query, key)) {
				r = m - 1;
			}
			else if (LT(key, query)) {
				l = m + 1;
			}
		}
		RowVectorX texture = textureList.row(sampleIndex[idx]);
#ifdef SHOW
		this->scaleShape(shape);
		Mesh mesh(shape);
		cv::Mat image = cv::Mat::zeros((int)(yScale * 2 + 10), (int)(xScale * 2 + 10), CV_8UC3);
		Texture::renderTexOnMesh(texture, mesh, image);
		mesh.drawMesh(image);
		cv::imshow("fit", image);
		cv::waitKey(10);
#endif
		return texture;
	}

	std::string TexFitModel::toString() const {
		std::string ret = "";
		ret += p_pcaShp->toString();
		ret += String::matToBytes(projectedShape);
		printf("Save projectShape %d %d\n", projectedShape.rows(), projectedShape.cols());
		ret += String::matToBytes(textureList);
		printf("Save textureList %d %d\n", textureList.rows(), textureList.cols());
		for (int i = 0; i < projectedShape.rows(); ++i) {
			ret += String::numToBytes<int>(sampleIndex[i]);
		}
		ret += String::numToBytes<float>(centerPoint.x);
		ret += String::numToBytes<float>(centerPoint.y);
		ret += String::numToBytes<float>(xScale);
		ret += String::numToBytes<float>(yScale);
		printf("Save centerPoint %f %f\n", centerPoint.x, centerPoint.y);
		printf("Save scale %f %f\n", xScale, yScale);
		return ret;
	}

	bool TexFitModel::fromStringStream(std::stringstream &str) {
		this->reset();
		p_pcaShp = new PCA();
		p_pcaShp->fromStringStream(str);
		String::matFromBytes(str, projectedShape);
		printf("Load projectShape %d %d\n", projectedShape.rows(), projectedShape.cols());
		String::matFromBytes(str, textureList);
		printf("Load textureList %d %d\n", textureList.rows(), textureList.cols());
		for (int i = 0; i < projectedShape.rows(); ++i) {
			sampleIndex.push_back(String::numFromBytes<int>(str));
		}
		String::numFromBytes<float>(str, centerPoint.x);
		String::numFromBytes<float>(str, centerPoint.y);
		String::numFromBytes<float>(str, xScale);
		String::numFromBytes<float>(str, yScale);
		printf("Load centerPoint %f %f\n", centerPoint.x, centerPoint.y);
		printf("Load scale %f %f\n", xScale, yScale);
		return true;
	}
}
