#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main()
{
    // Transform vertex position into light's orthogonal space
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}