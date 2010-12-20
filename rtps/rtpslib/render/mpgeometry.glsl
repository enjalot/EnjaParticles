#version 330
//#geometry shader

layout(triangles) in;
layout(triangle_strip) out;
out vec2 texCoord;

#define radius 0.01
#define layer 1

void main() 
{
    //for(int i = 0; i < gl_in.length(); i++) {  // avoid duplicate draw

    int j = 0;
    for (int j=0; j < gl_in.length(); j++) 
    {
        vec4 p = gl_in[0].gl_Position;
        texCoord = vec2(1.0,1.0);
        gl_Position = vec4(p.r+radius, p.g+radius+j*0.05, p.b, p.a);
        EmitVertex();

        texCoord = vec2(0.0,1.0);
        gl_Position = vec4(p.r-radius, p.g+radius+j*0.05, p.b, p.a);
        EmitVertex();

        texCoord = vec2(1.0,0.0);
        gl_Position = vec4(p.r+radius, p.g-radius+j*0.05, p.b, p.a);
        EmitVertex();

        texCoord = vec2(0.0,0.0);
        gl_Position = vec4(p.r-radius, p.g-radius+j*0.05, p.b, p.a);
        EmitVertex();

        EndPrimitive();
    }
//}
}

