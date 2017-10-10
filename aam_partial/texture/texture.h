#pragma once

#include <core/aam.h>
#include <geo/mesh.h>
#include <geo/procrustes.h>

namespace aam {
	class Texture {
	private:
		static int g_w, g_h;
		static Mesh g_meanMesh;
		static MatrixX g_texCoords;
	public:
		friend class Saver;
		static cv::Point2i round(const Point &p) {
			return cv::Point2i((int)std::round(p.x), (int)std::round(p.y));
		}
		static Color colorAt(const cv::Mat &image, int x, int y) {
			const unsigned char *ptr = image.ptr<unsigned char>(y);
			return cv::Scalar(ptr[x * 3], ptr[x * 3 + 1], ptr[x * 3 + 2]);
		}
		static Color bilinear(const Point &p, const cv::Mat &image);
		static MatrixX texCoordsInMeanMesh(Mesh &meanMesh);
		static MatrixX &getTexCoords() {
			assert(g_texCoords.rows() > 0);
			return g_texCoords;
		}
		// for pca

		template <typename T>
		static RowVectorX_<T> warpTexToMeanMesh(const RowVectorX &rawShape, const cv::Mat &rawImage) {
#ifdef SHOW
			cv::Mat mat = cv::Mat::zeros(g_h, g_w, CV_8UC3);
			const RowVectorX &meanShape = Procrustes::getMeanShape();
#endif
			RowVectorX_<T> ret(1, g_texCoords.rows() * 3);
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
					ret[pi * 3 + i] = (T)color[i];
				}
			}
#ifdef SHOW
			renderTexOnMesh(ret, g_meanMesh, mat);
			g_meanMesh.drawMesh(mat);
			cv::imshow("texOnMeanShape", mat);
#endif
			return ret;
		}

		template <typename T>
		static MatrixX_<T> collectTextures(const MatrixX &rawShapeList, const std::vector<std::string> &imgList) {
			assert(rawShapeList.rows() == imgList.size());
			MatrixX_<T> ret;
			int size = rawShapeList.rows(); if (size > MAX_SAMPLES) size = MAX_SAMPLES;
			for (int imgI = 0; imgI < size; ++imgI) {
				auto &vec = rawShapeList.row(imgI);
				cv::Mat mat = cv::imread(imgList[imgI]);
				g_w = mat.cols;
				g_h = mat.rows;
				auto texture = aam::Texture::warpTexToMeanMesh<T>(vec, mat);
				if (ret.rows() == 0) {
					ret.resize(size, texture.cols());
				}
				ret.row(imgI) = texture;
#ifdef SHOW
				cv::waitKey(10);
#endif
			}
			return ret;
		}
		template <typename T>
		static MatrixX_<T> collectTextures(const MatrixX &rawShapeList, cv::VideoCapture &cap) {
			//assert(rawShapeList.rows() == (int)cap.get(CV_CAP_PROP_FRAME_COUNT));
			cap.set(CV_CAP_PROP_POS_FRAMES, 0);
			cv::Mat image;
			MatrixX_<T> ret;
			int size = rawShapeList.rows(); if (size > MAX_SAMPLES) size = MAX_SAMPLES;
			for (int imgI = 0; imgI < size; ++imgI) {
				printf("%d\r", imgI);
				auto &vec = rawShapeList.row(imgI);
				cap >> image;
				g_w = image.cols;
				g_h = image.rows;
				auto texture = aam::Texture::warpTexToMeanMesh<T>(vec, image);
				if (ret.rows() == 0) {
					ret.resize(size, texture.cols());
				}
				ret.row(imgI) = texture;
#ifdef SHOW
				cv::waitKey(10);
#endif
			}
			return ret;
		}
		template <class T>
		static void renderTexOnMesh(const RowVectorX_<T> &texture, Mesh &mesh, cv::Mat &image) {
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
					Scalar vec[3];
					bool hit = mesh.isInside(Point((Scalar)x, (Scalar)y), tri, vec[0], vec[1], vec[2]);
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

		static void getResolution(int &w, int &h) {
			w = g_w;
			h = g_h;
		}
	};
}
