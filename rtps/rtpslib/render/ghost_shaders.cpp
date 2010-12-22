#define STRINGIFY(A) #A

//this code heavily based off NVIDIA's oclParticle example from the OpenCL SDK

const char* vertex_shader_source = STRINGIFY(

varying vec3 posEye;        // position of center in eye space

void main()
{

    posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    gl_PointSize = 20.0;
    
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    float a = gl_Vertex.w;
    gl_FrontColor = vec4(a, a, a, a);
    //gl_FrontColor = vec4(1, 1, 1, 1);

    
}


);

const char* fragment_shader_source = STRINGIFY(

uniform float pointRadius;  // point size in world space
//varying float pointRadius;  // point size in world space
varying vec3 posEye;        // position of center in eye space

void main()
{
    // calculate normal from texture coordinates
    vec3 n;
    n.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(n.xy, n.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    
    gl_FragColor = gl_Color;
}


);
