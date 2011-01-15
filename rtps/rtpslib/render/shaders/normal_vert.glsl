uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

void main()
{
	gl_PointSize = pointRadius;
	gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = ftransform();
}
