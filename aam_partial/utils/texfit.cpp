#include "texfit.h"
#include "print.h"
#define PCA_SORT
namespace aam {
	const int sortCnt = 30;
	const int sortOrder[sortCnt] = {
		12, 18, 13, 17, 25, 29,
		19, 21, 22, 24, 20, 23,
		15, 14, 16, 27, 26, 28, 9, 10, 11,
		0, 8, 6, 1, 7, 3, 5, 2, 4
	};
	void TexFitModel::normalizeShapes(MatrixX &shapeList) {
		// center
		centerPoint = Point(0, 0);
		const auto &meanShape = Procrustes::getMeanShape();
		Scalar min_x = meanShape[0], max_x = meanShape[0];
		Scalar min_y = meanShape[1], max_y = meanShape[1];
		for (int j = 0; j < meanShape.size(); j += 2) {
			Scalar x = meanShape[j];
			Scalar y = meanShape[j + 1];
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
				Scalar x = row[j];
				Scalar y = row[j + 1];
				x = x - centerPoint.x;
				y = y - centerPoint.y;
				if (std::abs(x) > xScale) xScale = std::abs(x);
				if (std::abs(y) > yScale) yScale = std::abs(y);
			}
		}
		for (int i = 0; i < shapeList.rows(); ++i) {
			auto &row = shapeList.row(i);
			for (int j = 0; j < row.cols(); j += 2) {
				Scalar x = row[j];
				Scalar y = row[j + 1];
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
		alignedShapeList = Procrustes::alignShapes(rawShapeList, 10);
		this->normalizeShapes(alignedShapeList);
		std::cout << "Triangulation\n";
		auto triangleIndexes = Triangulation::delaunayTriangulation(Procrustes::getMeanShape());
		auto meanMesh = Mesh(Procrustes::getMeanShape());
		auto texCoords = Texture::texCoordsInMeanMesh(meanMesh);
		std::cout << "Collect texture\n";
		textureList = Texture::collectTextures<byte>(rawShapeList, videoCapture);
		videoCapture.release();
		smoothTextures(30);
		std::cout << "Smooth texture\n";

		// pca on aligned shape
		std::cout << "PCA on shape\n";
		p_pcaShp = new PCA(alignedShapeList, 0.99f);
		std::cout << "Shape has components: " << p_pcaShp->getNumComponents() << std::endl;
		projectedShape = p_pcaShp->project(alignedShapeList);
		// sort samples
		for (int i = 0; i < alignedShapeList.rows(); ++i) {
			sampleIndex.push_back(i);
		}
		this->qsort(sampleIndex, 0, sampleIndex.size() - 1);
#ifdef SHOW
		system("pause");
		testFitTexture(alignedShapeList, false, false);
#endif
	}

#ifdef PCA_SORT
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
#else
	void TexFitModel::sort(std::vector<int> &idx, int l, int r) {
		for (int i = l; i <= r; ++i) {
			for (int j = i + 1; j <= r; ++j) {
				if (LT(alignedShapeList.row(idx[j]), alignedShapeList.row(idx[i]))) {
					int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
				}
			}
		}
	}

	void TexFitModel::qsort(std::vector<int> &idx, int l, int r) {
		int i = l, j = r;
		int m = (i + j) / 2;
		RowVectorX key(1, alignedShapeList.row(idx[m]).cols());
		key << alignedShapeList.row(idx[m]);
		while (i <= j) {
			while (LT(alignedShapeList.row(idx[i]), key) && i < r) ++i;
			while (LT(key, alignedShapeList.row(idx[j])) && l < j) --j;
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
			int k = (i >= sortCnt) ? (i - sortCnt) : i;
			k = k * 2 + (i >= sortCnt);
			if (X[k] > Y[k]) return false;
			if (X[k] < Y[k]) return true;
		}
		return false;
	}

	bool TexFitModel::EQ(const RowVectorX &X, const RowVectorX &Y) {
		for (int i = 0; i < X.size(); ++i) {
			int k = (i >= sortCnt) ? (i - sortCnt) : i;
			k = k * 2 + (i >= sortCnt);
			if (!(std::abs(X[k] - Y[k]) < 1e-6)) return false;
		}
		return true;
	}
#endif

	RowVectorX_<byte> TexFitModel::fitTexture(const RowVectorX &queryShape, bool isRawShape, bool needNormalized) {
		int idx = -1;
		// project queryShape
		RowVectorX shape = queryShape;
		if (isRawShape) Procrustes::procrustes(Procrustes::getMeanShape(), shape);
		if (isRawShape || needNormalized) normalizeShape(shape);
#ifdef PCA_SORT
		const auto &shapeList = projectedShape;
		auto query = p_pcaShp->project(shape);
#else
		const auto &shapeList = alignedShapeList;
		const auto &query = queryShape;
#endif
		if (idx < 0) {
			int l = 0, r = shapeList.rows() - 1;
			while (l <= r) {
				int m = (l + r) / 2;
				const auto & key = shapeList.row(sampleIndex[m]);
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
		}
		RowVectorX_<byte> texture = textureList.row(sampleIndex[idx]);
#ifdef SHOW
		printf("%d\r", idx);
		this->scaleShape(shape);
		Mesh mesh(shape);
		cv::Mat image = cv::Mat::zeros((int)(yScale * 2 + 10), (int)(xScale * 2 + 10), CV_8UC3);
		Texture::renderTexOnMesh<byte>(texture, mesh, image);
		mesh.drawMesh(image);
		cv::imshow("fit", image);
		cv::waitKey(10);
#endif
		return texture;
	}

	std::string TexFitModel::toString() const {
		std::string ret = "";
		ret += p_pcaShp->toString();
		ret += String::matToBytes<MatrixX, Scalar>(projectedShape);
		printf("Save projectShape %d %d\n", projectedShape.rows(), projectedShape.cols());
		ret += String::matToBytes<MatrixX, Scalar>(alignedShapeList);
		printf("Save alignedShapeList %d %d\n", alignedShapeList.rows(), alignedShapeList.cols());
		ret += String::matToBytes<MatrixX_<byte>, byte>(textureList);
		printf("Save textureList %d %d\n", textureList.rows(), textureList.cols());
		for (int i = 0; i < projectedShape.rows(); ++i) {
			ret += String::numToBytes<int>(sampleIndex[i]);
		}
		ret += String::numToBytes<Scalar>(centerPoint.x);
		ret += String::numToBytes<Scalar>(centerPoint.y);
		ret += String::numToBytes<Scalar>(xScale);
		ret += String::numToBytes<Scalar>(yScale);
		printf("Save centerPoint %f %f\n", centerPoint.x, centerPoint.y);
		printf("Save scale %f %f\n", xScale, yScale);
		return ret;
	}

	bool TexFitModel::fromistream(std::istream &str) {
		this->reset();
		p_pcaShp = new PCA();
		p_pcaShp->fromistream(str);
		String::matFromBytes<MatrixX, Scalar>(str, projectedShape);
		printf("Load projectShape %d %d\n", projectedShape.rows(), projectedShape.cols());
		String::matFromBytes<MatrixX, Scalar>(str, alignedShapeList);
		printf("Load alignedShapeList %d %d\n", alignedShapeList.rows(), alignedShapeList.cols());
		String::matFromBytes<MatrixX_<byte>, byte>(str, textureList);
		printf("Load textureList %d %d\n", textureList.rows(), textureList.cols());
		for (int i = 0; i < projectedShape.rows(); ++i) {
			sampleIndex.push_back(String::numFromBytes<int>(str));
		}
		String::numFromBytes<Scalar>(str, centerPoint.x);
		String::numFromBytes<Scalar>(str, centerPoint.y);
		String::numFromBytes<Scalar>(str, xScale);
		String::numFromBytes<Scalar>(str, yScale);
		printf("Load centerPoint %f %f\n", centerPoint.x, centerPoint.y);
		printf("Load scale %f %f\n", xScale, yScale);
		return true;
	}

	void TexFitModel::smoothTextures(int step) {
		MatrixX_<byte> newList(textureList.rows(), textureList.cols());
		std::cout << "new texture " << newList.rows() << " " << newList.cols() << std::endl;
		for (int i = 0; i < textureList.rows(); ++i) {
			int cnt = 0;
			int l = i - step / 2;
			int r = l + step;
			l = std::max(0, l);
			r = std::min(r, (int)textureList.rows());
			std::vector<int> newRow(textureList.cols());
			for (int k = 0; k < newRow.size(); ++k) newRow[k] = 0;
			for (int j = l; j < r; ++j) {
				cnt += 1;
				for (int k = 0; k < newRow.size(); ++k)
					newRow[k] += (int)textureList.row(j).col(k).value();
			}
			for (int k = 0; k < newRow.size(); ++k) {
				newRow[k] = (int)((float)newRow[k] / (float)cnt);
				newList(i, k) = (byte)newRow[k];
			}
		}
		std::cout << "new texture " << newList.rows() << " " << newList.cols() << std::endl;
		textureList = newList;
	}
}
