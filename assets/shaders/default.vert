#version 450

layout(location = 0) out vec3 vertex_color;

void main() {

  const vec3 positions[3] = vec3[3](
      vec3(0.5f, 0.5f, 0.0f),  //
      vec3(-0.5f, 0.5f, 0.0f), //
      vec3(0.f, -0.5f, 0.0f)   //
  );

  const vec3 colors[3] = vec3[3](
      vec3(0.f, 1.f, 1.0f), //
      vec3(1.f, 1.f, 0.0f), //
      vec3(1.f, 0.f, 1.0f)  //
  );

  vertex_color   = colors[gl_VertexIndex];
  gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
}

