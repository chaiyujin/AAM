#include <Python.h>
#include <numpy/arrayobject.h>
#include "pybind.h"
#include <iostream>

aam::TexFitModel g_model;
bool g_inited  = false;

static PyObject *
aam_buildAndSave(PyObject *self, PyObject *args) {
	const char *videoPath;
	const char *savePath;
	if (!PyArg_ParseTuple(args, "zz", &videoPath, &savePath))
		return NULL;

	printf("Building aam...\n");
	g_model.buildFromVideo(videoPath);
	printf("Save aam...\n");
	aam::Saver::save(savePath, g_model);
	printf("Building and saving are done!\n");
	g_inited = true;

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
aam_load(PyObject *self, PyObject *args) {
	const char *savePath;
	if (!PyArg_ParseTuple(args, "z", &savePath))
		return NULL;

	printf("Load aam...\n");
	aam::Saver::load(savePath, g_model);
	g_inited = true;
	printf("Loading is done!\n");

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
aam_normalizeShape(PyObject *self, PyObject *args) {
	assert(g_inited);
	PyObject *arg = NULL;
	PyObject *arr = NULL;
	aam::RowVectorX row;
	std::vector<float> data;
	if (!PyArg_ParseTuple(args, "O", &arg))
		return NULL;

	arr = PyArray_FROM_OTF(arg, NPY_FLOAT, NPY_IN_ARRAY);
	if (arr == NULL)
		return NULL;

	int nd = PyArray_NDIM(arr);   //number of dimensions
	npy_intp *dims = PyArray_DIMS(arr);
	if (nd > 2 ||
		(nd == 2 && dims[1] != 2) /* (x, y) */) {
		goto fail;
	}
	float *dptr = (float *)PyArray_DATA(arr);
	// flatten
	int idx = 0;
	for (int d = 0; d < nd; ++d) {
		for (int i = 0; i < dims[d]; ++i) {
			data.push_back(dptr[idx++]);
		}
	}
	// normalize
	row.resize(1, data.size());
	for (int i = 0; i < data.size(); ++i) row[i] = (float)data[i];
	aam::Procrustes::procrustes(aam::Procrustes::getMeanShape(), row); // align
	g_model.normalizeShape(row); // normalize
	// return
	{
		int *o_dims = new int[nd];
		for (int i = 0; i < nd; ++i) o_dims[i] = dims[i];
		PyArrayObject *output = (PyArrayObject *)PyArray_FromDims(
			nd, o_dims, NPY_FLOAT
		);
		float *out = (float *)output->data;
		for (int i = 0; i < row.size(); ++i) {
			out[i] = (float)row[i];
		}
		Py_DECREF(arr);
		return PyArray_Return(output);
	}

fail:
	Py_XDECREF(arr);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
aam_normalizeLandmarks(PyObject *self, PyObject *args) {
	assert(g_inited);
	PyObject *arg = NULL;
	PyObject *arr = NULL;
	aam::RowVectorX row;
	std::vector<float> data;
	if (!PyArg_ParseTuple(args, "O", &arg))
		return NULL;

	arr = PyArray_FROM_OTF(arg, NPY_FLOAT, NPY_IN_ARRAY);
	if (arr == NULL)
		return NULL;

	int nd = PyArray_NDIM(arr);   //number of dimensions
	npy_intp *dims = PyArray_DIMS(arr);
	if (nd > 2 ||
		(nd == 2 && dims[1] != 2) /* (x, y) */) {
		goto fail;
	}
	float *dptr = (float *)PyArray_DATA(arr);
	// flatten
	int idx = 0;
	for (int d = 0; d < nd; ++d) {
		for (int i = 0; i < dims[d]; ++i) {
			data.push_back(dptr[idx++]);
		}
	}
	// normalize
	row.resize(1, data.size());
	int g_w, g_h;
	aam::Texture::getResolution(g_w, g_h);
	g_w /= 2;
	g_h /= 2;
	for (int i = 0; i < data.size(); i += 2) {
		row[i] = (float)data[i] * g_w + g_w;
		row[i + 1] = (float)data[i + 1] * g_h + g_h;
	}
	aam::Procrustes::procrustes(aam::Procrustes::getMeanShape(), row); // align
	g_model.normalizeShape(row); // normalize
								 // return
	{
		int *o_dims = new int[nd];
		for (int i = 0; i < nd; ++i) o_dims[i] = dims[i];
		PyArrayObject *output = (PyArrayObject *)PyArray_FromDims(
			nd, o_dims, NPY_FLOAT
		);
		float *out = (float *)output->data;
		for (int i = 0; i < row.size(); ++i) {
			out[i] = (float)row[i];
		}
		Py_DECREF(arr);
		return PyArray_Return(output);
	}

fail:
	Py_XDECREF(arr);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
aam_renderNormShape(PyObject *self, PyObject *args) {
	assert(g_inited);
	PyObject *arg = NULL;
	PyObject *arr = NULL;
	aam::RowVectorX row;
	aam::RowVectorXByte texture;
	std::vector<float> data;
	if (!PyArg_ParseTuple(args, "O", &arg))
		return NULL;

	arr = PyArray_FROM_OTF(arg, NPY_FLOAT, NPY_IN_ARRAY);
	if (arr == NULL)
		return NULL;

	int nd = PyArray_NDIM(arr);   //number of dimensions
	npy_intp *dims = PyArray_DIMS(arr);
	if (nd > 2 ||
		(nd == 2 && dims[1] != 2) /* (x, y) */) {
		Py_XDECREF(arr);
		Py_INCREF(Py_None);
		return Py_None;
	}
	float *dptr = (float *)PyArray_DATA(arr);
	// flatten
	int idx = 0;
	for (int d = 0; d < nd; ++d) {
		for (int i = 0; i < dims[d]; ++i) {
			data.push_back(dptr[idx++]);
		}
	}
	// copy
	row.resize(1, data.size());
	for (int i = 0; i < data.size(); ++i) {
		row[i] = (float)data[i];
	}
	// texture
	texture = g_model.fitTexture(row, false, false);
	g_model.scaleShape(row);
	aam::Mesh mesh(row);
	// scale
	float x, y;
	g_model.getScale(x, y);
	x = x * 2 + 10;
	y = y * 2 + 10;
	cv::Mat image = cv::Mat::zeros((int)y, (int)x, CV_8UC3);
	aam::Texture::renderTexOnMesh(texture, mesh, image);
	mesh.drawMesh(image);
	{
		int *o_dims = new int[3];
		o_dims[0] = image.rows;
		o_dims[1] = image.cols;
		o_dims[2] = 3;
		PyArrayObject *output = (PyArrayObject *)PyArray_FromDims(
			3, o_dims, NPY_UINT8
		);
		uint8_t *out = (uint8_t *)output->data;
		int idx = 0;
		for (int i = 0; i < image.rows; ++i) {
			uint8_t *ptr = image.ptr<uint8_t>(i);
			for (int j = 0; j < image.cols; ++j) {
				for (int k = 0; k < 3; ++k) {
					out[idx++] = ptr[j * 3 + k];
				}
			}
		}
		Py_DECREF(arr);
		return PyArray_Return(output);
	}

fail:
	Py_XDECREF(arr);
	Py_INCREF(Py_None);
	return Py_None;
}


static PyMethodDef Methods[] = {
	{
		"build_and_save", aam_buildAndSave,
		METH_VARARGS,
		"build and save"
	},
	{
		"load", aam_load,
		METH_VARARGS,
		"load"
	},
	{
		"normalize_shape", aam_normalizeShape,
		METH_VARARGS,
		"normalize shape"
	},
	{
		"normalize_landmarks", aam_normalizeLandmarks,
		METH_VARARGS,
		"normalize landmarks"
	},
	{
		"render_norm_shape", aam_renderNormShape,
		METH_VARARGS,
		"render normalized shape"
	},
	{ NULL, NULL, 0, NULL }        /* Sentinel */
};

static struct PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	"aam",   /* name of module */
	NULL, /* module documentation, may be NULL */
	-1,       /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
	Methods
};

PyMODINIT_FUNC
PyInit_aam(void) {
	import_array();
	return PyModule_Create(&module);
}
