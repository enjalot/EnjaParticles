uniform float pointRadius;  // point size in world space
uniform float near;
uniform float far;

uniform sampler2D col;      //texture to draw on the sprite

varying vec3 posEye;        // position of center in eye space

void main()
{
    vec4 color = texture2D(col, gl_TexCoord[0].st);
    //gl_FragColor = color * vec4(1.0, 1.0, 1.0, gl_Color.a);
    gl_FragColor = vec4(1.) - gl_Color + color;
}
