import { useMemo } from 'react';
import { StreamResponseType, ProcessStep } from '../types';

interface ProcessStepsProps {
  streamingContent: any[];
}

export default function ProcessSteps({ streamingContent }: ProcessStepsProps) {
  const steps = useMemo(() => {
    const stepStatus = new Map<ProcessStep, {
      status: 'pending' | 'in-progress' | 'completed' | 'error',
      message?: string,
      llmOutput?: string
    }>();

    streamingContent.forEach((content) => {
      if (content.step && content.type) {
        const currentStep = stepStatus.get(content.step) || { 
          status: 'pending', 
          message: '', 
          llmOutput: '' 
        };

        switch (content.type) {
          case StreamResponseType.STEP_START:
            stepStatus.set(content.step, { 
              ...currentStep,
              status: 'in-progress', 
              message: content.message 
            });
            break;
          case StreamResponseType.STEP_END:
            stepStatus.set(content.step, { 
              ...currentStep,
              status: 'completed', 
              message: content.message 
            });
            break;
          case StreamResponseType.ERROR:
            stepStatus.set(content.step, { 
              ...currentStep,
              status: 'error', 
              message: content.content 
            });
            break;
          case StreamResponseType.LLM_CHUNK:
            stepStatus.set(content.step, {
              ...currentStep,
              llmOutput: (currentStep.llmOutput || '') + content.content
            });
            break;
        }
      }
    });

    return stepStatus;
  }, [streamingContent]);

  return (
    <div className="mb-6">
      <h3 className="text-lg font-medium mb-4">Process Steps</h3>
      <div className="space-y-2">
        {Array.from(steps.entries()).map(([step, { status, message, llmOutput }]) => (
          <div
            key={step}
            className={`p-3 rounded-md bg-gray-800 text-white`}
          >
            <div className="flex items-center">
              <span className="font-medium">{step}</span>
              <span className="ml-2 text-sm">
                {status === 'error' ? '❌' : status === 'completed' ? '✅' : '⏳'}
              </span>
            </div>
            {message && <p className="text-sm mt-1">{message}</p>}
            {llmOutput && (
              <div className="mt-2 text-sm bg-gray-700 p-2 rounded">
                {llmOutput}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
