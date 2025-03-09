import { useMemo } from 'react';
import { StreamResponseType } from '../types';

interface QueryResultsProps {
  streamingContent: any[];
}

export default function QueryResults({ streamingContent }: QueryResultsProps) {
  const {
    sql,
    explanation,
    results,
    llmOutput
  } = useMemo(() => {
    let sql = '';
    let explanation = '';
    let results: any = null;
    let llmOutput = new Map<string, string>();

    streamingContent.forEach((content) => {
      if (content.type === StreamResponseType.LLM_CHUNK) {
        // Store LLM output by step
        const stepOutput = llmOutput.get(content.step) || '';
        llmOutput.set(content.step, stepOutput + content.content);

        // Also update specific sections
        if (content.step === 'SQL') {
          sql += content.content;
        } else if (content.step === 'EXPLANATION') {
          explanation += content.content;
        }
      } else if (content.type === StreamResponseType.RESULT) {
        if (content.step === 'EXECUTION') {
          results = content.content;
        }
      }
    });

    return { 
      sql, 
      explanation, 
      results, 
      llmOutput: Object.fromEntries(llmOutput) 
    };
  }, [streamingContent]);

  return (
    <div className="space-y-6">
      {sql && (
        <div className="bg-gray-50 p-4 rounded-md">
          <h4 className="font-medium mb-2">Generated SQL</h4>
          <pre className="bg-gray-800 text-white p-4 rounded overflow-x-auto">
            {sql}
          </pre>
        </div>
      )}

      {results && (
        <div className="bg-gray-50 p-4 rounded-md">
          <h4 className="font-medium mb-2">Query Results</h4>
          <div className="overflow-x-auto">
            <pre className="bg-gray-800 text-white p-4 rounded">
              {JSON.stringify(results, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {explanation && (
        <div className="bg-gray-50 p-4 rounded-md">
          <h4 className="font-medium mb-2">Explanation</h4>
          <div className="prose max-w-none whitespace-pre-wrap">
            {explanation}
          </div>
        </div>
      )}
    </div>
  );
}
