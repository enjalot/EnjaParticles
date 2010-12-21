//these shaders are taken straight from Mike Pan's blog post:
// http://blog.mikepan.com/some-more-shader/


int length;
std::string path(GLSL_SOURCE_DIR);
path += "/mpvertex.glsl";
const char* vertex_shader_source = file_contents(path.c_str(), &length);


path = GLSL_SOURCE_DIR;
path += "/mpgeometry.glsl";
const char* geometry_shader_source = file_contents(path.c_str(), &length);


path = GLSL_SOURCE_DIR;
path += "/mpfragment.glsl";
const char* fragment_shader_source = file_contents(path.c_str(), &length);


/*

#define STRINGIFY(A) #A
//==========================
const char* vertex_shader_source = STRINGIFY(

#version 120
#vertex shader

in vec4 vertex;
uniform float timer;

void main() 
{
    //vec4 v = vertex;
    vec4 v = gl_Vertex;
    v.z = sin(timer+v.y+v.x)*0.5+v.z;
    v.x = sin(timer*10.0+v.y+v.y)*0.2+v.x;
    gl_Position = gl_ModelViewProjectionMatrix * v;
}

);

const char* geometry_shader_source = STRINGIFY(

//==========================
#version 330
#geometry shader


layout(triangles) in;
layout(triangle_strip) out;
varying vec2 texCoord;

##define radius 0.01
##define layer 1
//float radius = 0.01;
//int layer = 1;

void main() 
{
    //for(int i = 0; i < gl_in.length(); i++) {  // avoid duplicate draw

    for (int j=0; j vec4 p = gl_in[0].gl_Position;)
    {
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
);
const char* fragment_shader_source = STRINGIFY(

//========================================
//version 330
//fragment shader

in vec2 texCoord;
out vec4 outColor;

uniform sampler2D col;
uniform float emit
uniform float alpha;

void main(void) 
{

    // load color texture
    vec4 color;
    color = texture2D(col, texCoord);

    // apply material panel values
    color.rgb *= emit;
    color.a *= alpha;

    outColor = color;
}
);
*/
