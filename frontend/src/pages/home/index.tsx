import React, { useState }  from 'react';
import { Layout, Typography } from 'antd';
import SearchBox from '../../components/SearchBox';
import ImageGrid from '../../components/ImageGrid';
import styles from './index.module.css';

import type { ImageResult } from '../../types';

const { Content } = Layout;
const { Title } = Typography;

const Home: React.FC = () => {
  const [searchResults, setSearchResults] = useState<Map<string, ImageResult[]>>()

  const handleSearchResults = (results: Map<string, ImageResult[]>) => {
    setSearchResults(results);
  }

  return (
    <Layout className={styles.home}>
      <Title level={2} className={styles.title}>
        PersonaLens VisionHub
      </Title>
      <Content className={styles.content}>
        <SearchBox onSearchResults={handleSearchResults} />
        <ImageGrid searchResults={searchResults} />
      </Content>
    </Layout>
  );
};

export default Home;