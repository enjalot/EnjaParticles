uniform float pointRadius;  // point size in world space
varying vec3 posEye;        // position of center in eye space
void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);
    const float shininess = 40.0;

    // calculate normal from texture coordinates
    vec3 n;
    n.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(n.xy, n.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    n.z = sqrt(1.0-mag);

    // point on surface of sphere in eye space
    vec4 spherePosEye = vec4(posEye + n*pointRadius,1.0);
	vec4 clipSpacePos = gl_ProjectionMatrix*spherePosEye;
	gl_FragDepth = clipSpacePos.z/clipSpacePos.w;

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, n));
    /*vec3 v = normalize(-spherePosEye);
    vec3 h = normalize(lightDir + v);
    float specular = pow(max(0.0, dot(n, h)), shininess);*/

	//gl_FragColor.rgb = vec3(gl_FragDepth);
    gl_FragColor = gl_Color * diffuse;
    //gl_FragColor = gl_Color * diffuse + specular;
}
