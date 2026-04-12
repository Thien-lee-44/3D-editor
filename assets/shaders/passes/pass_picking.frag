#version 330 core

out vec4 FragColor;
uniform vec3 u_ColorId;

void main() {
    // Encodes the unique Entity ID as a flat RGB color
    FragColor = vec4(u_ColorId, 1.0);
}