import axios from 'axios';

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:4523/api/v1'
).replace(/\/$/, '');
export const fetchImages = async (query: string, model: string) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/images_text_retrieval`, {
      search: query,
      model: model,
    },
    {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 1000 * 600000
    }
    );
    if (response.status !== 200) {
      throw new Error(`请求失败，状态码: ${response.status}`);
    }
    console.log('响应数据:', response.data);
    return response.data;
  } catch (error) {
    console.error('无法进行图文检索:', error);
    throw error;
  }
};

export const fetchImagesAllModels = async (query: string, models: string[]) => {
  try {
    const results = new Map();
    let gallerySize = 0

    for (const model of models) {
      try {
        console.log(`正在请求模型: ${model}`);
        const response = await fetchImages(query, model);
        results.set(model, response);
        gallerySize = Math.max(gallerySize, response.gallery_size);
        console.log(`模型 ${model} 请求完成，当前已完成 ${results.size}/${models.length}`);
      } catch (error) {
        console.error(`模型 ${model} 搜索失败:`, error);
      }
    }

    console.log(`所有请求完成，成功获取 ${results.size} 个模型的数据`);

    return {
      results: results,
      gallerySize: gallerySize
    }
  } catch (error) {
    console.error('批量搜索失败:', error);
    throw error;
  }
};