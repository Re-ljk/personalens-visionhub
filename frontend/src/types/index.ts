export interface ImageResult {
  id: number;
  score: number;
  pic: string; // Assuming pic is a base64 encoded string or URL
  is_query: boolean;
}