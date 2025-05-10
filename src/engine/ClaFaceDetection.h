
#pragma once

#include "FaceSuper.h"

//. Image Position of Face detection Information 
struct ST_FaceRectInfo
{
	int		m_nTop;

	int		m_nBottom;

	int		m_nRight;

	int		m_nLeft;
};

//. Common Class of Face Detection
class ClaFaceDetection
{
public:
	ClaFaceDetection();

	~ClaFaceDetection();

public:
	void	Init();

	void	SetParam();

};