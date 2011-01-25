
void main()
{
	gl_TexCoord[0] = gl_MultiTexCoord0;
    //vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = ftransform();
}
