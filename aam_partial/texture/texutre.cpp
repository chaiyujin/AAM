#include "texture.h"
#include <geo/triangulation.h>
#include <geo/procrustes.h>
#include <geo/bbox.h>
#include <vector>

namespace aam {
	int Texture::g_h;
	int Texture::g_w;
	MatrixX Texture::g_texCoords;
	Mesh Texture::g_meanMesh;
	MatrixX Texture::texCoordsInMeanMesh(Mesh &meanMesh) {
		g_meanMesh = meanMesh;
		BBox bbox = meanMesh.getBBox();
		std::vector<std::vector<Scalar>> coordsList;
		for (int y = (int)bbox.startY(); y <= (int)bbox.endY(); ++y) {
			for (int x = (int)bbox.startX(); x <= (int)bbox.endX(); ++x) {
				int tri;
				float a, b, c;
				bool hit = meanMesh.isInside(Point((Scalar)x, (Scalar)y), tri, a, b, c);
				// hit!
				if (hit) {
					std::vector<Scalar> coords;
					coords.push_back((Scalar)tri);
					coords.push_back(a);
					coords.push_back(b);
					coords.push_back(c);
					coordsList.push_back(coords);
				}
			}
		}
		g_texCoords.resize(coordsList.size(), coordsList[0].size());
		for (int i = 0; i < g_texCoords.rows(); ++i) {
			g_texCoords.row(i) = Eigen::Matrix<Scalar, 4, 1>(coordsList[i][0], coordsList[i][1], coordsList[i][2], coordsList[i][3]);
		}
		return g_texCoords;
	}


	RowVectorX Texture::warpTexToMeanMesh(const RowVectorX &rawShape, const cv::Mat &rawImage) {
#ifdef SHOW
		cv::Mat mat = cv::Mat::zeros(720, 1280, CV_8UC3);
		const RowVectorX &meanShape = Procrustes::getMeanShape();
#endif
		RowVectorX ret(1, g_texCoords.rows() * 3);
		const auto &trIndexes = Triangulation::getTriangleIndexes();
		for (int pi = 0; pi < g_texCoords.rows(); ++pi) {
			const auto &vec = g_texCoords.row(pi);
			int tri = (int)vec[0];
			// point index
			int v[3];
			for (int i = 0; i < 3; ++i) v[i] = trIndexes[tri * 3 + i];
			Scalar rawX = 0, rawY = 0;
			for (int i = 0; i < 3; ++i) {
				rawX += rawShape[v[i] * 2] * vec[i + 1];
				rawY += rawShape[v[i] * 2 + 1] * vec[i + 1];
			}
			Color color = bilinear(Point(rawX, rawY), rawImage);
			for (int i = 0; i < 3; ++i) {
				ret[pi * 3 + i] = color[i];
			}
		}
#ifdef SHOW
		renderTexOnMesh(ret, g_meanMesh, mat);
		g_meanMesh.drawMesh(mat);
		cv::imshow("texOnMeanShape", mat);
#endif
		return ret;
	}

	void Texture::renderTexOnMesh(const RowVectorX &texture, Mesh &mesh, cv::Mat &image) {
		cv::Mat mean = cv::Mat::zeros(g_h, g_w, image.type());
		const auto &trIndexes = Triangulation::getTriangleIndexes();
		const auto &shape = Procrustes::getMeanShape();
		// on mean mesh
		for (int pi = 0; pi < g_texCoords.rows(); ++pi) {
			const auto &vec = g_texCoords.row(pi);
			int tri = (int)vec[0];
			// point index
			int v[3];
			for (int i = 0; i < 3; ++i) v[i] = trIndexes[tri * 3 + i];
			Scalar x = 0, y = 0;
			for (int i = 0; i < 3; ++i) {
				x += shape[v[i] * 2] * vec[i + 1];
				y += shape[v[i] * 2 + 1] * vec[i + 1];
			}
			int xi = (int)std::round(x) * 3;
			int yi = (int)std::round(y);
			unsigned char *ptr = mean.ptr<unsigned char>(yi);
			for (int i = 0; i < 3; ++i) {
				unsigned char v = (unsigned char)texture[pi * 3 + i];
				if (texture[pi * 3 + i] > 255) v = 255;
				else if (texture[pi * 3 + i] < 0) v = 0;
				ptr[xi + i] = v;
			}
		}
		// warp to mesh
		BBox bbox = mesh.getBBox();
		for (int y = (int)bbox.startY(); y <= (int)bbox.endY(); ++y) {
			for (int x = (int)bbox.startX(); x <= (int)bbox.endX(); ++x) {
				int tri;
				float vec[3];
				bool hit = mesh.isInside(Point(x, y), tri, vec[0], vec[1], vec[2]);
				// hit!
				if (hit) {
					// draw
					int v[3];
					for (int i = 0; i < 3; ++i) v[i] = trIndexes[tri * 3 + i];
					Scalar rawX = 0, rawY = 0;
					for (int i = 0; i < 3; ++i) {
						rawX += shape[v[i] * 2] * vec[i];
						rawY += shape[v[i] * 2 + 1] * vec[i];
					}
					Color color = bilinear(Point(rawX, rawY), mean);
					unsigned char *ptr = image.ptr<unsigned char>(y);
					for (int i = 0; i < 3; ++i) {
						ptr[x * 3 + i] = (unsigned char)color[i];
					}
				}
			}
		}
	}

	Color Texture::bilinear(const Point &p, const cv::Mat &image) {
		int x = (int)p.x;
		int y = (int)p.y;

		float rx = p.x - x;
		float ry = p.y - y;
		Color c00 = colorAt(image, x, y);
		Color c01 = colorAt(image, x, y + 1);
		Color c10 = colorAt(image, x + 1, y);
		Color c11 = colorAt(image, x + 1, y + 1);
		Color c0 = c00 * (1 - ry) + c01 * ry;
		Color c1 = c10 * (1 - ry) + c11 * ry;
		return c0 * (1 - rx) + c1 * rx;
	}

	MatrixX Texture::collectTextures(const MatrixX &rawShapeList, const std::vector<std::string> &imgList) {
		assert(rawShapeList.rows() == imgList.size());
		MatrixX ret;
		for (int imgI = 0; imgI < imgList.size(); ++imgI) {
			auto &vec = rawShapeList.row(imgI);
			cv::Mat mat = cv::imread(imgList[imgI]);
			g_w = mat.cols;
			g_h = mat.rows;
			auto texture = aam::Texture::warpTexToMeanMesh(vec, mat);
			if (ret.rows() == 0) {
				ret.resize(imgList.size(), texture.cols());
			}
			ret.row(imgI) = texture;
#ifdef SHOW
			cv::waitKey(10);
#endif
		}
		return ret;
	}


	MatrixX Texture::collectTextures(const MatrixX &rawShapeList, cv::VideoCapture &cap) {
		//assert(rawShapeList.rows() == (int)cap.get(CV_CAP_PROP_FRAME_COUNT));
		cap.set(CV_CAP_PROP_POS_FRAMES, 0);
		cv::Mat image;
		MatrixX ret;
		for (int imgI = 0; imgI < rawShapeList.rows(); ++imgI) {
			auto &vec = rawShapeList.row(imgI);
			cap >> image;
			g_w = image.cols;
			g_h = image.rows;
			auto texture = aam::Texture::warpTexToMeanMesh(vec, image);
			if (ret.rows() == 0) {
				ret.resize(rawShapeList.rows(), texture.cols());
			}
			ret.row(imgI) = texture;
#ifdef SHOW
			cv::waitKey(10);
#endif
		}
		return ret;
	}
}