//varying float w;
void main()
{
	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
	//w = (gl_ModelViewProjectionMatrix * gl_Vertex).w;
    //gl_Position = ftransform();
}
