import React from 'react';
import { Card, Row, Col, Typography, Image, Divider } from 'antd';
import type { ImageResult } from '../../types';
import styles from './index.module.css';

const { Title, Text } = Typography;

interface ImageGridProps {
  searchResults: Map<string, ImageResult[]> | null | undefined;
}

const ImageGrid: React.FC<ImageGridProps> = ({ searchResults }) => {
  if (!searchResults || searchResults.size === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyState}>
          <Text type="secondary">暂无搜索结果</Text>
        </div>
      </div>
    );
  }

  // 处理 base64 图片数据
  const getImageSrc = (base64Data: string) => {
    // 如果已经包含 data URL 前缀，直接返回
    if (base64Data.startsWith('data:')) {
      return base64Data;
    }
    // 否则添加 data URL 前缀
    return `data:image/jpeg;base64,${base64Data}`;
  };

  const getImageTypeDescription = (isQuery: boolean) => {
    return isQuery ? "类型：原图" : "类型：相似图";
  };

  return (
    <div className={styles.container}>
      {Array.from(searchResults.entries()).map(([modelName, images]) => (
        <div key={modelName} className={styles.modelSection}>
          {/* 标题居中显示 */}
          <div className={styles.modelTitleContainer}>
            <Title level={3} className={styles.modelTitle}>
              {modelName}
            </Title>
          </div>
          {/* 图片容器占满全宽 */}
          <div className={styles.imagesContainer}>
            <Row gutter={[8, 8]} className={styles.imageRow}>
              {images.map((imageData, index) => (
                <Col xs={12} sm={8} md={6} lg={4} xl={3} xxl={2} key={`${modelName}-${imageData.id}-${index}`}>
                  <div className={imageData.is_query ? styles.highlightWrapper : styles.normalWrapper}>
                    <Card
                      hoverable
                      className={styles.imageCard}
                      cover={
                        <div className={styles.imageContainer}>
                          <Image
                            src={getImageSrc(imageData.pic)}
                            alt={`Image ${imageData.id}`}
                            className={styles.image}
                            preview={{
                              mask: '预览图片'
                            }}
                            onError={() => console.error(`Failed to load image: ${imageData.id}`)}
                          />
                        </div>
                      }
                    >
                      <Card.Meta
                        title={
                          <div className={styles.cardMeta}>
                            <Text strong>图片ID: {imageData.id}</Text>
                          </div>
                        }
                        description={
                          <div className={styles.cardMeta}>
                            <div style={{ marginBottom: '4px' }}>
                              <Text type="secondary">
                                {getImageTypeDescription(imageData.is_query || false)}
                              </Text>
                            </div>
                            { !imageData.is_query && (
                              <div>
                                <Text type="secondary">
                                  概率: {(imageData.score * 100).toFixed(1)}%
                                </Text>
                              </div>
                            )}
                            { !imageData.is_query && (
                              <div>
                                <Text 
                                  type="secondary"
                                  style={{ 
                                    color: '#1890ff',
                                    fontWeight: 'bold',
                                    fontSize: '12px'
                                  }}
                                >
                                  Top-{index}
                                </Text>
                              </div>
                            )}
                          </div>
                        }
                      />
                    </Card>
                  </div>
                </Col>
              ))}
            </Row>
          </div>
          <Divider />
        </div>
      ))}
      
    </div>
  );
};

export default ImageGrid;