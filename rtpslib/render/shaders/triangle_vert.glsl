void main(void)
{
   gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
   gl_FrontColor = gl_Color;
}
