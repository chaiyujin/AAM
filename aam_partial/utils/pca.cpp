#include "pca.h"

namespace aam {

	PCA::PCA(const MatrixX &samplesOnRows, float percentage) {
		// center aligned
		mean = samplesOnRows.colwise().mean();
		auto aligned = samplesOnRows.rowwise() - mean;
		Eigen::JacobiSVD<MatrixX> svd(aligned, Eigen::ComputeThinV);
		eigenValues = svd.singularValues();
		float *sum = new float[eigenValues.size()];
		sum[0] = eigenValues[0];
		for (int i = 1; i < eigenValues.size(); ++i) {
			sum[i] = sum[i - 1] + eigenValues[i];
		}
		numComponents = eigenValues.size();
		for (int i = 0; i < eigenValues.size(); ++i) {
			if (sum[i] / sum[eigenValues.size() - 1] >= percentage) {
				numComponents = i + 1;
				break;
			}
		}
		delete sum;
		W = svd.matrixV().leftCols(numComponents);
	}
}