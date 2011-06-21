#version 330

precision highp float;
layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 3) out;

void main(void)
{
    int i;

    for (i = 0; i<gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position+gl_in[i].gl_PointSize*vec4(1.0,0.0,0.0,0.0);
        EmitVertex();
        gl_Position = gl_in[i].gl_Position+gl_in[i].gl_PointSize*vec4(-.5,-0.5,0.0,0.0);
        EmitVertex();
        gl_Position = gl_in[i].gl_Position+gl_in[i].gl_PointSize*vec4(-0.5,0.5,0.0,0.0);
        EmitVertex();
        EndPrimitive();
    }
}
