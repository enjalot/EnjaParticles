#version 330
#fragment shader
in vec2 texCoord;
out vec4 outColor;

uniform sampler2D col;
uniform float emit, alpha;

void main(void) {

    // load color texture
    vec4 color;
    color = texture2D(col, texCoord);

    // apply material panel values
    color.rgb *= emit;
    color.a *= alpha;

    outColor = color;
}
