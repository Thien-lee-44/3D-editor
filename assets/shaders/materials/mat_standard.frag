#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec3 VertColor;
in vec4 FragPosLightSpace;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 emission;
    float shininess;
    float opacity;
};

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

#define MAX_DIR_LIGHTS 8
#define MAX_POINT_LIGHTS 16
#define MAX_SPOT_LIGHTS 8

uniform int numDirLights;
uniform int numPointLights;
uniform int numSpotLights;

uniform DirLight dirLights[MAX_DIR_LIGHTS];
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform SpotLight spotLights[MAX_SPOT_LIGHTS];

uniform vec3 viewPos;
uniform mat4 view;  
uniform Material material;

uniform int combLight;
uniform int combTex;
uniform int combVColor;

uniform sampler2D mapDiffuse;   uniform int hasMapDiffuse;
uniform sampler2D mapSpecular;  uniform int hasMapSpecular;
uniform sampler2D mapAmbient;   uniform int hasMapAmbient;
uniform sampler2D mapEmission;  uniform int hasMapEmission;
uniform sampler2D mapShininess; uniform int hasMapShininess;
uniform sampler2D mapOpacity;   uniform int hasMapOpacity;
uniform sampler2D mapBump;      uniform int hasMapBump;
uniform sampler2D mapReflection;uniform int hasMapReflection;

uniform sampler2D shadowMap;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0 || projCoords.z < 0.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) return 0.0;
    
    float currentDepth = projCoords.z;
    vec3 n = normalize(normal);
    vec3 l = normalize(-lightDir);
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float ndotl = max(dot(n, l), 0.0);
    //float slopeBias = 0.00035 * (1.0 - ndotl);
    //float texelBias = 0.75 * max(texelSize.x, texelSize.y);
    //float bias = clamp(max(0.00008 + slopeBias, texelBias), 0.00008, 0.00075);
    float slope = sqrt(max(1.0 - ndotl * ndotl, 0.0)) / max(ndotl, 0.001);
    float bias = clamp(0.00005 + 0.0001 * slope, 0.0003, 0.001);
    float shadow = 0.0;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    return shadow;
}

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 diffColor, vec3 specColor, vec3 ambientColor, float shine, float shadow) {
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shine);
    
    vec3 ambient = light.ambient * ambientColor;
    vec3 diffuse = light.diffuse * diff * diffColor;
    vec3 specular = light.specular * spec * specColor;
    
    return ambient + (1.0 - shadow) * (diffuse + specular);
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 diffColor, vec3 specColor, vec3 ambientColor, float shine) {
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shine);
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    vec3 ambient = light.ambient * ambientColor;
    vec3 diffuse = light.diffuse * diff * diffColor;
    vec3 specular = light.specular * spec * specColor;
    return (ambient + diffuse + specular) * attenuation;
}

vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 diffColor, vec3 specColor, vec3 ambientColor, float shine) {
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shine);
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    vec3 ambient = light.ambient * ambientColor;
    vec3 diffuse = light.diffuse * diff * diffColor;
    vec3 specular = light.specular * spec * specColor;
    return (ambient + diffuse + specular) * attenuation * intensity;
}

vec3 getNormalFromMap(vec3 baseNormal, vec3 fragPos, vec2 texCoords) {
    vec3 tangentNormal = texture(mapBump, texCoords).xyz * 2.0 - 1.0;
    vec3 Q1  = dFdx(fragPos);
    vec3 Q2  = dFdy(fragPos);
    vec2 st1 = dFdx(texCoords);
    vec2 st2 = dFdy(texCoords);
    vec3 T_vec = Q1 * st2.y - Q2 * st1.y;
    
    if (length(T_vec) <= 0.0) return normalize(baseNormal);
    float det = (st1.x * st2.y - st2.x * st1.y);
    float detSign = det < 0.0 ? -1.0 : 1.0;
    vec3 T = normalize(T_vec) * detSign;
    
    vec3 N = normalize(baseNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * tangentNormal);
}

void main() {
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 baseColor = material.diffuse;
    vec3 ambientBase = material.ambient;
    vec3 specColor = material.specular;
    vec3 emisColor = material.emission;
    float shine = max(material.shininess, 0.0001); 
    float alpha = material.opacity;
    vec3 reflColor = vec3(0.0);

    if (combVColor == 1) {
        baseColor *= VertColor;
        ambientBase *= VertColor;
    }

    if (combTex == 1) {
        if (hasMapOpacity == 1) alpha *= texture(mapOpacity, TexCoords).r;
        if (alpha < 0.01) discard; 

        if (hasMapDiffuse == 1) {
            vec4 tex = texture(mapDiffuse, TexCoords);
            baseColor *= tex.rgb;
            ambientBase *= tex.rgb;
            alpha *= tex.a; 
        }
        if (alpha < 0.01) discard;
        
        if (hasMapBump == 1) norm = getNormalFromMap(norm, FragPos, TexCoords);
        if (hasMapSpecular == 1) specColor *= texture(mapSpecular, TexCoords).rgb;
        if (hasMapEmission == 1) emisColor += texture(mapEmission, TexCoords).rgb;
        if (hasMapAmbient == 1)  ambientBase *= texture(mapAmbient, TexCoords).rgb; 
        
        if (hasMapShininess == 1) {
            shine = max(shine * texture(mapShininess, TexCoords).r * 128.0, 0.0001);
        }
        
        if (hasMapReflection == 1) {
            vec3 I = normalize(FragPos - viewPos);
            vec3 R = reflect(I, norm);
            vec3 viewR = mat3(view) * R;
            float m = 2.0 * sqrt(pow(viewR.x, 2.0) + pow(viewR.y, 2.0) + pow(viewR.z + 1.0, 2.0));
            m = max(m, 0.0001);
            vec2 sphereUV = viewR.xy / m + 0.5;
            reflColor = texture(mapReflection, sphereUV).rgb;
        }
    }

    vec3 result = vec3(0.0);
    
    if (combLight == 1) {
        for(int i = 0; i < numDirLights; i++) {
            float shadow = (i == 0) ? ShadowCalculation(FragPosLightSpace, norm, dirLights[i].direction) : 0.0;
            result += CalcDirLight(dirLights[i], norm, viewDir, baseColor, specColor, ambientBase, shine, shadow);
        }
        for(int i = 0; i < numPointLights; i++)
            result += CalcPointLight(pointLights[i], norm, FragPos, viewDir, baseColor, specColor, ambientBase, shine);
        for(int i = 0; i < numSpotLights; i++)
            result += CalcSpotLight(spotLights[i], norm, FragPos, viewDir, baseColor, specColor, ambientBase, shine);
            
        result += reflColor * specColor;
    } else {
        result = baseColor + reflColor * specColor;
    }
    
    result += emisColor;
    FragColor = vec4(result, alpha);
}