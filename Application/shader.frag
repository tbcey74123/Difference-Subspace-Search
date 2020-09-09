#version 330 core
layout(location = 0) out vec4 fragColor;

uniform vec4 diffuse, ambient;
struct VertexData {
	vec3 normal;
	vec3 lightDir1;
	vec3 lightDir2;
	vec3 lightDir3;
	vec3 eyeVec;
};
in VertexData vertexData;

void main() {
	vec4 final_color = ambient;

    vec3 N = normalize(vertexData.normal);
    vec3 L1 = normalize(vertexData.lightDir1);
    float lambertTerm1 = dot(N, L1);
    if(lambertTerm1 > 0.0)
    {
		final_color += diffuse * lambertTerm1 * 0.6;
    }
    vec3 L2 = normalize(vertexData.lightDir2);
    float lambertTerm2 = dot(N, L2);
    if(lambertTerm2 > 0.0)
    {
		final_color += diffuse * lambertTerm2 * 0.3;
    }
    vec3 L3 = normalize(vertexData.lightDir3);
    float lambertTerm3 = dot(N, L3);
    if(lambertTerm3 > 0.0)
    {
		final_color += diffuse * lambertTerm3 * 0.1;
    }
    fragColor = final_color;
}