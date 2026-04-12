#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 VertColor;

struct Material {
    vec3 diffuse;
    vec3 emission;
    float opacity;
};

uniform Material material;
uniform int combTex;
uniform int combVColor;

uniform sampler2D mapDiffuse;   uniform int hasMapDiffuse;
uniform sampler2D mapOpacity;   uniform int hasMapOpacity;
uniform sampler2D mapEmission;  uniform int hasMapEmission;

void main() {
    vec3 baseColor = material.diffuse;
    vec3 emisColor = material.emission;
    float alpha = material.opacity;

    if (combVColor == 1) {
        baseColor *= VertColor;
    }

    if (combTex == 1) {
        if (hasMapOpacity == 1) alpha *= texture(mapOpacity, TexCoords).r;
        if (alpha < 0.01) discard; 

        if (hasMapDiffuse == 1) {
            vec4 tex = texture(mapDiffuse, TexCoords);
            baseColor *= tex.rgb;
            alpha *= tex.a;
        }
        if (alpha < 0.01) discard;
        
        if (hasMapEmission == 1) emisColor += texture(mapEmission, TexCoords).rgb;
    }

    FragColor = vec4(baseColor + emisColor, alpha);
}