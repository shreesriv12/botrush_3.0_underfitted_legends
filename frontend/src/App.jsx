import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [safetyImage, setSafetyImage] = useState(null);
  const [pathImage, setPathImage] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setError(null);
    
    // Create preview of selected image
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setProcessing(true);
    setResult(null);
    setSafetyImage(null);
    setPathImage(null);
    setError(null);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('image', selectedFile);

      // Send request to backend
      const response = await axios.post('http://localhost:5000/process_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResult(response.data.result);
      
      // Get the safety and path images if available
      if (response.data.safety_image_url) {
        setSafetyImage(`http://localhost:5000${response.data.safety_image_url}`);
      }
      
      if (response.data.path_image_url) {
        setPathImage(`http://localhost:5000${response.data.path_image_url}`);
      }
    } catch (err) {
      console.error('Error processing image:', err);
      setError(err.response?.data?.error || 'Error processing image');
    } finally {
      setProcessing(false);
    }
  };

  // Render a visual representation of the safety matrix
  const renderMatrix = (matrix) => {
    if (!matrix) return null;
    
    return (
      <div className="flex flex-col border border-gray-300 rounded overflow-hidden">
        {matrix.map((row, rowIndex) => (
          <div key={`row-${rowIndex}`} className="flex">
            {row.map((cell, cellIndex) => (
              <div 
                key={`cell-${rowIndex}-${cellIndex}`}
                className={`w-8 h-8 flex items-center justify-center border border-gray-200 ${
                  cell === 'S' || cell === 1 ? 'bg-green-200' : 'bg-red-200'
                }`}
              >
                {cell}
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white py-6 shadow-md">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold">Grid Analysis and Pathfinding</h1>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4">Upload Grid Image</h2>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <input 
                type="file" 
                onChange={handleFileChange} 
                accept="image/*"
                id="file-input"
                className="hidden"
              />
              <label 
                htmlFor="file-input" 
                className="block w-full py-2 px-4 cursor-pointer bg-gray-200 hover:bg-gray-300 rounded text-center transition-colors"
              >
                {selectedFile ? selectedFile.name : 'Choose an image'}
              </label>
            </div>
            
            <button 
              type="submit" 
              className={`w-full py-2 px-4 rounded font-medium text-white ${
                !selectedFile || processing 
                  ? 'bg-blue-400 cursor-not-allowed' 
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
              disabled={!selectedFile || processing}
            >
              {processing ? 'Processing...' : 'Analyze Image'}
            </button>
          </form>
          
          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              {error}
            </div>
          )}
          
          {preview && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-2">Preview</h3>
              <img 
                src={preview} 
                alt="Preview" 
                className="max-w-full h-auto max-h-64 rounded border border-gray-300" 
              />
            </div>
          )}
        </div>

        {processing && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-8 text-center">
            <div className="inline-block w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
            <p className="text-gray-600">Processing image. This may take a moment...</p>
          </div>
        )}
        
        {result && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-6">Analysis Results</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="border rounded-lg p-4">
                <h3 className="text-xl font-semibold mb-4">Safety Classification</h3>
                {safetyImage && (
                  <img 
                    src={safetyImage} 
                    alt="Safety Analysis" 
                    className="max-w-full h-auto rounded mb-4" 
                  />
                )}
                <div className="mt-4">
                  <h4 className="text-lg font-medium mb-2">Safety Matrix</h4>
                  {renderMatrix(result.safety_matrix)}
                </div>
              </div>
              
              <div className="border rounded-lg p-4">
                <h3 className="text-xl font-semibold mb-4">Path Finding</h3>
                {pathImage ? (
                  <img 
                    src={pathImage} 
                    alt="Path Finding" 
                    className="max-w-full h-auto rounded mb-4" 
                  />
                ) : (
                  <div className="py-8 text-center text-gray-600">
                    {result.path_exists === false 
                      ? "No valid path found!" 
                      : "Path visualization not available"
                    }
                  </div>
                )}
                
                {result.path && (
                  <div className="mt-4">
                    <h4 className="text-lg font-medium mb-2">Path Coordinates</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.path.map((coord, idx) => (
                        <span 
                          key={`path-${idx}`} 
                          className="inline-block px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm"
                        >
                          {`(${coord[0]},${coord[1]})`}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;