#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#define SHOW

namespace aam {
	typedef cv::Point2d Point;
	typedef cv::Scalar Color;
	typedef double Scalar;
	template<class Scalar, int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
	struct MatrixTraits {
		typedef Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor> MatrixType;
		typedef Eigen::Map<MatrixType, 0, Eigen::Stride<Eigen::Dynamic, 1> > MatrixMapType;
		typedef Eigen::Map<MatrixType const, 0, Eigen::Stride<Eigen::Dynamic, 1> > ConstMatrixMapType;
		typedef Eigen::Ref<MatrixType> MatrixRefType;
		typedef Eigen::Ref<MatrixType const> ConstMatrixRefType;
	};
	typedef MatrixTraits<Scalar,	1, Eigen::Dynamic>::MatrixType	RowVectorX;
	typedef MatrixTraits<int,		1, Eigen::Dynamic>::MatrixType	RowVectorXi;
	typedef MatrixTraits<Scalar,	1, 2>::MatrixType				RowVector2;
	typedef MatrixTraits<Scalar,	1, 3>::MatrixType				RowVector3;
	typedef MatrixTraits<Scalar			>::MatrixType				MatrixX;
	typedef MatrixTraits<Scalar,	2, 2>::MatrixType				Matrix2;

	// view
	template<class Scalar>
	inline typename MatrixTraits<Scalar>::MatrixMapType
		toSeparatedView(typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixRefType m, int dims = 2)
	{
		return typename MatrixTraits<Scalar>::MatrixMapType(m.data(), m.cols() / dims, dims, Eigen::Stride<Eigen::Dynamic, 1>(dims, 1));
	}

	template<class Scalar>
	inline typename MatrixTraits<Scalar>::ConstMatrixMapType
		toSeparatedViewConst(typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixRefType m, int dims = 2)
	{
		return typename MatrixTraits<Scalar>::ConstMatrixMapType(m.data(), m.cols() / dims, dims, Eigen::Stride<Eigen::Dynamic, 1>(dims, 1));
	}

	template<class Scalar>
	inline typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixMapType
		toInterleavedView(typename MatrixTraits<Scalar>::MatrixRefType m)
	{
		return typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::MatrixMapType(m.data(), 1, m.rows() * m.cols(), Eigen::Stride<Eigen::Dynamic, 1>(m.rows() * m.cols(), 1));
	}

	template<class Scalar>
	inline typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixMapType
		toInterleavedViewConst(typename MatrixTraits<Scalar>::ConstMatrixRefType m)
	{
		return typename MatrixTraits<Scalar, 1, Eigen::Dynamic>::ConstMatrixMapType(m.data(), 1, m.rows() * m.cols(), Eigen::Stride<Eigen::Dynamic, 1>(m.rows() * m.cols(), 1));
	}

}