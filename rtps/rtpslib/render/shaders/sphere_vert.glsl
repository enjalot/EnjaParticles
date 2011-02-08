uniform float pointRadius;
uniform float pointScale;   // scale to calculate size in pixels
//uniform bool blending;
varying vec3 posEye;        // position of center in eye space


void main()
{

    posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
    //gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0);
	//gl_Position = ftransform();

    gl_FrontColor = gl_Color;
}
