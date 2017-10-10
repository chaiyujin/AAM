#include "model.h"

namespace aam {

	void Model::normalizeShapes(MatrixX &shapeList) {
		// center
		centerPoint = Point(0, 0);
		const auto &meanShape = Procrustes::getMeanShape();
		for (int j = 0; j < meanShape.size(); j += 2) {
			float x = meanShape[j];
			float y = meanShape[j + 1];
			centerPoint.x += x;
			centerPoint.y += y;
		}
		centerPoint.x /= (meanShape.size() / 2.f);
		centerPoint.y /= (meanShape.size() / 2.f);
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

	void Model::buildFromVideo(std::string videoPath) {
		this->reset();
		MatrixX rawShapeList;
		cv::VideoCapture videoCapture;
		std::cout << "Load video and lms from " << videoPath << std::endl;
		LMSUtil::loadFromVideo(videoPath, rawShapeList, videoCapture);
		std::cout << rawShapeList.rows() << " " << rawShapeList.cols() << std::endl;
		std::cout << "Align and normalize shapes\n";
		auto alignedShapeList = Procrustes::alignShapes(rawShapeList, 10);
		this->normalizeShapes(alignedShapeList);
		std::cout << "Triangulation\n";
		auto triangleIndexes = Triangulation::delaunayTriangulation(Procrustes::getMeanShape());
		auto meanMesh = Mesh(Procrustes::getMeanShape());
		auto texCoords = Texture::texCoordsInMeanMesh(meanMesh);
		std::cout << "Collect texture\n";
		auto textureList = Texture::collectTextures<Scalar>(rawShapeList, videoCapture);

		// release video
		videoCapture.release();

		// real build
		build(alignedShapeList, textureList);

		// build over
		m_isGood = true;
	}

	void Model::buildFromDir(std::string dirPath) {
		this->reset();
		MatrixX rawShapeList;
		std::vector<std::string> imgList;
		std::cout << "Load image and asf from dir " << dirPath << std::endl;
		ASFUtil::loadFromDir(dirPath, rawShapeList, imgList);
		std::cout << "Align and normalize shapes\n";
		auto alignedShapeList = Procrustes::alignShapes(rawShapeList, 10);
		this->normalizeShapes(alignedShapeList);
		std::cout << "Triangulation\n";
		auto triangleIndexes = Triangulation::delaunayTriangulation(Procrustes::getMeanShape());
		auto meanMesh = Mesh(Procrustes::getMeanShape());
		auto texCoords = Texture::texCoordsInMeanMesh(meanMesh);
		std::cout << "Collect texture\n";
		auto textureList = Texture::collectTextures<Scalar>(rawShapeList, imgList);
		
		// real build
		build(alignedShapeList, textureList);

		// build over
		m_isGood = true;
	}

	void Model::build(MatrixX &alignedShapeList, MatrixX &textureList) {
		// pca on aligned shape
		std::cout << "PCA on shape\n";
		p_pcaShp = new PCA(alignedShapeList, 0.99f);
		std::cout << "Shape has components: " << p_pcaShp->getNumComponents() << std::endl;
		float shapeEnergy = p_pcaShp->getEigenValues().sum();
		std::cout << "Shape energy: " << shapeEnergy << std::endl;
		// pca on texture on mean shape
		std::cout << "PCA on texture\n";
		p_pcaTex = new PCA(textureList, 0.99f);
		std::cout << "Texture has components: " << p_pcaTex->getNumComponents() << std::endl;
		float textureEnergy = p_pcaTex->getEigenValues().sum();
		std::cout << "Texture energy: " << textureEnergy << std::endl;
		// pca on combined
		// texture Scale
		textureScale = std::sqrt(shapeEnergy / textureEnergy);
		std::cout << "Texture weight: " << textureScale << std::endl;
		// combine
		auto projectedShp = p_pcaShp->project<MatrixX>(alignedShapeList);
		auto porjectedTex = p_pcaTex->project<MatrixX>(textureList);
		MatrixX combined(projectedShp.rows(), projectedShp.cols() + porjectedTex.cols());
		combined << projectedShp, porjectedTex * textureScale;
		std::cout << "PCA on appearance\n";
		p_pcaApp = new PCA(combined, 0.99f);
		std::cout << "Appearance has components: " << p_pcaApp->getNumComponents() << std::endl;
#ifdef SHOW
		system("pause");
#endif
		// mesh list for rendering
		std::vector<Mesh> alignedMeshList;
		for (int i = 0; i < alignedShapeList.rows(); ++i) {
			alignedMeshList.emplace_back(alignedShapeList.row(i));
		}

		std::cout << "Show texture\n";
		for (int i = 0; i < alignedShapeList.rows(); ++i) {
			auto app = this->combine<RowVectorX>(alignedShapeList.row(i), textureList.row(i));
			RowVectorX texture, shape;
			this->seperate<RowVectorX>(app, shape, texture);
			this->scaleShape(shape);
			cv::Mat image = cv::Mat::zeros(720, 1280, CV_8UC3);
			Mesh mesh(shape);
			Texture::renderTexOnMesh<Scalar>(texture, mesh, image);
			mesh.drawMesh(image);
			cv::imshow("texture", image);
			cv::waitKey(30);
		}
	}
}