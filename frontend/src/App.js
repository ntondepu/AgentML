import React, { useState, useEffect } from 'react';
import { Layout, Menu, Card, Row, Col, Statistic, Badge, Button, Space, Tabs } from 'antd';
import { 
  DashboardOutlined, 
  RobotOutlined, 
  ClusterOutlined, 
  MessageOutlined,
  BarChartOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';
import './App.css';

const { Header, Content, Sider } = Layout;
const { TabPane } = Tabs;

// Mock data - in real app, this would come from API
const mockData = {
  mlPipelines: [
    { id: 1, name: 'Customer Churn Prediction', status: 'running', progress: 75 },
    { id: 2, name: 'Fraud Detection', status: 'completed', progress: 100 },
    { id: 3, name: 'Price Optimization', status: 'pending', progress: 0 }
  ],
  distributedNodes: [
    { id: 'node_1', role: 'leader', status: 'running' },
    { id: 'node_2', role: 'follower', status: 'running' },
    { id: 'node_3', role: 'follower', status: 'running' },
    { id: 'node_4', role: 'follower', status: 'running' },
    { id: 'node_5', role: 'follower', status: 'stopped' }
  ],
  chatSessions: [
    { id: 1, title: 'ML Pipeline Help', messages: 8, active: true },
    { id: 2, title: 'Distributed System Query', messages: 12, active: false }
  ]
};

function App() {
  const [selectedKey, setSelectedKey] = useState('dashboard');
  const [data, setData] = useState(mockData);
  const [raftNodes, setRaftNodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { sender: 'assistant', text: 'Hi! How can I help you with your ML pipeline or distributed system today?' }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    // In real app, fetch data from API
    const interval = setInterval(() => {
      // Simulate data updates
      setData(prev => ({
        ...prev,
        mlPipelines: prev.mlPipelines.map(pipeline => 
          pipeline.status === 'running' 
            ? { ...pipeline, progress: Math.min(100, pipeline.progress + Math.random() * 5) }
            : pipeline
        )
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    async function fetchRaftState() {
      setLoading(true);
      try {
        const res = await fetch('/api/raft/cluster-state');
        const data = await res.json();
        setRaftNodes(data.nodes || []);
      } catch (e) {
        setRaftNodes([]);
      }
      setLoading(false);
    }
    fetchRaftState();
    const interval = setInterval(fetchRaftState, 2000);
    return () => clearInterval(interval);
  }, []);

  const triggerElection = async () => {
    await fetch('/api/raft/trigger-election', { method: 'POST' });
  };
  const createPartition = async () => {
    await fetch('/api/raft/create-partition', { method: 'POST' });
  };
  const resetSimulation = async () => {
    await fetch('/api/raft/reset', { method: 'POST' });
  };

  const sendMessage = async () => {
    if (!chatInput.trim()) return;
    const userMsg = { sender: 'user', text: chatInput };
    setChatMessages(msgs => [...msgs, userMsg]);
    setChatLoading(true);
    try {
      const res = await fetch('http://localhost:8003/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user: 'user', message: chatInput })
      });
      const data = await res.json();
      setChatMessages(msgs => [...msgs, { sender: 'assistant', text: data.response }]);
    } catch (e) {
      setChatMessages(msgs => [...msgs, { sender: 'assistant', text: 'Sorry, there was an error contacting the chatbot.' }]);
    }
    setChatInput("");
    setChatLoading(false);
  };

  const menuItems = [
    { key: 'dashboard', icon: <DashboardOutlined />, label: 'Dashboard' },
    { key: 'ml-pipeline', icon: <RobotOutlined />, label: 'ML Pipeline' },
    { key: 'distributed', icon: <ClusterOutlined />, label: 'Distributed Sim' },
    { key: 'chatbot', icon: <MessageOutlined />, label: 'AI Chatbot' },
    { key: 'monitoring', icon: <BarChartOutlined />, label: 'Monitoring' },
    { key: 'settings', icon: <SettingOutlined />, label: 'Settings' }
  ];

  const renderDashboard = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Active ML Pipelines"
              value={data.mlPipelines.filter(p => p.status === 'running').length}
              prefix={<RobotOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Distributed Nodes"
              value={`${data.distributedNodes.filter(n => n.status === 'running').length}/${data.distributedNodes.length}`}
              prefix={<ClusterOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Chat Sessions"
              value={data.chatSessions.filter(s => s.active).length}
              prefix={<MessageOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="System Health"
              value="98.5%"
              suffix="%"
              prefix={<BarChartOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="ML Pipeline Status" extra={<Button icon={<SyncOutlined />}>Refresh</Button>}>
            <div className="pipeline-list">
              {data.mlPipelines.map(pipeline => (
                <div key={pipeline.id} className="pipeline-item">
                  <div className="pipeline-header">
                    <span>{pipeline.name}</span>
                    <Badge 
                      status={pipeline.status === 'running' ? 'processing' : 
                              pipeline.status === 'completed' ? 'success' : 'default'} 
                      text={pipeline.status}
                    />
                  </div>
                  <div className="pipeline-progress">
                    <div 
                      className="progress-bar"
                      style={{ width: `${pipeline.progress}%` }}
                    />
                    <span className="progress-text">{pipeline.progress}%</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Distributed Cluster" extra={<Button icon={<PlayCircleOutlined />}>Start Simulation</Button>}>
            <div className="node-grid">
              {data.distributedNodes.map(node => (
                <div key={node.id} className={`node-item ${node.status}`}>
                  <div className="node-header">
                    <span>{node.id}</span>
                    {node.role === 'leader' && <Badge status="success" text="Leader" />}
                  </div>
                  <div className="node-status">
                    <Badge 
                      status={node.status === 'running' ? 'success' : 'error'} 
                      text={node.status}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );

  const renderMLPipeline = () => (
    <div>
      <Card title="ML Pipeline Management" extra={<Button type="primary">Create Pipeline</Button>}>
        <Tabs defaultActiveKey="active">
          <TabPane tab="Active Pipelines" key="active">
            <div className="pipeline-detailed-list">
              {data.mlPipelines.filter(p => p.status === 'running').map(pipeline => (
                <Card key={pipeline.id} className="pipeline-card">
                  <h3>{pipeline.name}</h3>
                  <p>Status: <Badge status="processing" text={pipeline.status} /></p>
                  <p>Progress: {pipeline.progress}%</p>
                  <Space>
                    <Button>View Logs</Button>
                    <Button>Metrics</Button>
                    <Button danger>Stop</Button>
                  </Space>
                </Card>
              ))}
            </div>
          </TabPane>
          <TabPane tab="Completed" key="completed">
            <div className="pipeline-detailed-list">
              {data.mlPipelines.filter(p => p.status === 'completed').map(pipeline => (
                <Card key={pipeline.id} className="pipeline-card">
                  <h3>{pipeline.name}</h3>
                  <p>Status: <Badge status="success" text={pipeline.status} /></p>
                  <p>Progress: {pipeline.progress}%</p>
                  <Space>
                    <Button>View Results</Button>
                    <Button>Deploy</Button>
                    <Button>Download</Button>
                  </Space>
                </Card>
              ))}
            </div>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );

  const renderDistributedSim = () => (
    <div>
      <Card title="Distributed Systems Simulation" extra={<Button type="primary" onClick={triggerElection}>Trigger Election</Button>}>
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={16}>
            <Card title="Cluster Visualization" className="cluster-viz">
              <div className="cluster-diagram">
                {loading ? <div>Loading...</div> :
                  raftNodes.map(node => (
                    <div key={node.node_id} className={`cluster-node ${node.status} ${node.role.toLowerCase()}`}
                      style={{
                        background: node.role === 'LEADER' ? '#ffe58f' : node.status === 'RUNNING' ? '#e6f7ff' : '#fff1f0',
                        border: node.role === 'LEADER' ? '2px solid #faad14' : '1px solid #d9d9d9',
                        margin: 8, padding: 12, borderRadius: 8, minWidth: 100, textAlign: 'center'
                      }}>
                      <div className="node-label">{node.node_id}</div>
                      <div className="node-role">{node.role}</div>
                      <div className="node-status">{node.status}</div>
                      {node.is_partitioned && <Badge status="error" text="Partitioned" />}
                    </div>
                  ))
                }
              </div>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card title="Controls">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button block icon={<PlayCircleOutlined />} onClick={triggerElection}>Trigger Election</Button>
                <Button block icon={<PauseCircleOutlined />} onClick={createPartition}>Create Partition</Button>
                <Button block icon={<SyncOutlined />} onClick={resetSimulation}>Reset Simulation</Button>
              </Space>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );

  const renderChatbot = () => (
    <div>
      <Card title="AI Chatbot Interface">
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={16}>
            <Card title="Chat" className="chat-interface">
              <div className="chat-messages" style={{ minHeight: 200, maxHeight: 400, overflowY: 'auto', marginBottom: 16 }}>
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.sender}`}
                    style={{ textAlign: msg.sender === 'user' ? 'right' : 'left', margin: '8px 0' }}>
                    <span style={{ background: msg.sender === 'user' ? '#e6f7ff' : '#f6ffed', padding: 8, borderRadius: 8 }}>{msg.text}</span>
                  </div>
                ))}
              </div>
              <div className="chat-input" style={{ display: 'flex', gap: 8 }}>
                <input
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') sendMessage(); }}
                  placeholder="Type your message..."
                  style={{ flex: 1, padding: 8, borderRadius: 4, border: '1px solid #d9d9d9' }}
                  disabled={chatLoading}
                />
                <Button type="primary" onClick={sendMessage} loading={chatLoading}>Send</Button>
              </div>
            </Card>
          </Col>
          <Col xs={24} lg={8}>
            <Card title="Sessions">
              <div className="session-list">
                <div className="session-item">
                  <h4>Current Session</h4>
                  <p>{chatMessages.length} messages</p>
                  <Badge status="success" />
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );

  const renderContent = () => {
    switch (selectedKey) {
      case 'dashboard':
        return renderDashboard();
      case 'ml-pipeline':
        return renderMLPipeline();
      case 'distributed':
        return renderDistributedSim();
      case 'chatbot':
        return renderChatbot();
      default:
        return <div>Coming soon...</div>;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header className="header">
        <div className="logo">
          <h2 style={{ color: 'white', margin: 0 }}>AutoML Platform</h2>
        </div>
      </Header>
      <Layout>
        <Sider width={200} className="site-layout-background">
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            style={{ height: '100%', borderRight: 0 }}
            items={menuItems}
            onClick={({ key }) => setSelectedKey(key)}
          />
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content
            className="site-layout-background"
            style={{
              padding: 24,
              margin: 0,
              minHeight: 280,
            }}
          >
            {renderContent()}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
}

export default App;
