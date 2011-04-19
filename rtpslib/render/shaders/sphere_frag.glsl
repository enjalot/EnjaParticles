#version 120
uniform float pointRadius;  // point size in world space
uniform float near;
uniform float far;
varying vec3 posEye;        // position of center in eye space

void main()
{
    //const vec3 lightDir = vec3(0.577, 0.577, 0.577);
    //const float shininess = 50.0;

    // calculate normal from texture coordinates
    vec3 n;
    //we should find a better way of doing this...
    n.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(n.xy, n.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    n.z = sqrt(1.0-mag);

    // point on surface of sphere in eye space
    vec4 spherePosEye =vec4(posEye+n*pointRadius,1.0);

    vec4 clipSpacePos = gl_ProjectionMatrix*spherePosEye;
    float normDepth = clipSpacePos.z/clipSpacePos.w;

    // Transform into window coordinates coordinates
    gl_FragDepth = (((far-near)/2.)*normDepth)+((far+near)/2.);

    //This should be an input parameter.
    /*const vec3 lightDir = vec3(-0.5, 1.5, 5.);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, n));

    vec3 v = normalize(-spherePosEye.xyz);
    vec3 h = normalize(lightDir + v);
    float specular = pow(max(0.0, dot(n, h)), shininess); */

    gl_FragData[0] = gl_Color;//*vec4(1.0,1.0,1.0,0.1);//vec4(vec3(1.0)-gl_Color.rgb,gl_Color.a); //Thickness rendering
	//gl_FragData[0] = vec4(1.0,.0,.0,0.1);//Save the color
    //gl_FragData[0] = gl_Color * diffuse + specular;
}
