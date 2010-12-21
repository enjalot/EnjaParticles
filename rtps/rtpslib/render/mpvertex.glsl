#version 120

in vec4 vertex;
uniform float timer;

void main() 
{
    vec4 v = vertex;
    //vec4 v = gl_Vertex;
//    v.z = sin(timer+v.y+v.x)*0.5+v.z;
//    v.x = sin(timer*10.0+v.y+v.y)*0.2+v.x;
    gl_Position = gl_ModelViewProjectionMatrix * v;
}


