#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

uniform mat4 um4mv;
uniform mat4 um4p;

uniform vec4 main_light_dir;
uniform vec4 sub_light_dir;
uniform vec4 back_light_dir;

struct VertexData {
	vec3 normal;
	vec3 lightDir1;
	vec3 lightDir2;
	vec3 lightDir3;
	vec3 eyeVec;
};
out VertexData vertexData;

void main()
{
	vertexData.normal = (transpose(inverse(um4mv)) * vec4(normal, 1)).xyz;
	vec3 vLight1 = vec3(um4mv * main_light_dir);
	vec3 vLight2 = vec3(um4mv * sub_light_dir);
	vec3 vLight3 = vec3(um4mv * back_light_dir);
	vec3 vVertex = vec3(um4mv * vec4(pos, 1));
	vertexData.lightDir1 = vLight1;
	vertexData.lightDir2 = vLight2;
	vertexData.lightDir3 = vLight3;
	vertexData.eyeVec = -vVertex;
    gl_Position = um4p * um4mv * vec4(pos, 1);
}